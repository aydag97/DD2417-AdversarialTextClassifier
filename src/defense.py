import json
import os
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import TinyStoriesTokenizer
from model     import TransformerClassifier
from load_model import load_model, predict

from train import (
    BLOCK_SIZE,
    BATCH_SIZE,
    DEVICE,
    set_seed,
    pad_sequence,
    evaluate,
)


CHECKPOINT_IN  = "checkpoints/baseline_2layer_512.pt"
CHECKPOINT_OUT = "checkpoints/defended_2layer_512.pt"

ADV_EXAMPLES_PATH = "outputs/adversarial_examples.json"


DEFENSE_EPOCHS = 3

DEFENSE_LR = 1e-4

CLEAN_SAMPLE_SIZE = 2000

ADV_OVERSAMPLE = 2


def load_tokenizer_from_checkpoint(ckpt_path: str):
    ckpt     = torch.load(ckpt_path, map_location="cpu")
    tok_path = ckpt.get("tokenizer_path", "data/imdb_tokenizer.json")
    return TinyStoriesTokenizer.load(tok_path)


def load_model_from_checkpoint(ckpt_path: str, tokenizer):
    ckpt   = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"]

    model = TransformerClassifier(
        vocab_size = config["vocab_size"],
        block_size = config["block_size"],
        vector_dim = config["vector_dim"],
        n_heads    = config["n_heads"],
        n_layers   = config["n_layers"],
        hidden_dim = config["hidden_dim"],
        dropout    = 0.2,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    return model


def encode_sample(text: str, label: int, tokenizer) -> dict:
    _, ids         = tokenizer.tokenize(text)
    ids, attn_mask = pad_sequence(ids, BLOCK_SIZE)
    return {
        "input_ids":      torch.tensor(ids),
        "attention_mask": torch.tensor(attn_mask),
        "label":          torch.tensor(label),
    }


def encode_dataset_list(pairs: list, tokenizer) -> list:
    return [encode_sample(p["text"], p["label"], tokenizer) for p in pairs]


def clean_text(text: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_augmented_dataset(adv_examples: list, tokenizer, clean_n: int = CLEAN_SAMPLE_SIZE):
    

    adv_pairs          = [{"text": ex["text"], "label": ex["label"]} for ex in adv_examples]
    adv_pairs_repeated = adv_pairs * ADV_OVERSAMPLE

    print(f"  Loading {clean_n} clean IMDB training samples...")
    ds = load_dataset("stanfordnlp/imdb")["train"].shuffle(seed=42)
    clean_pairs = [
        {"text": clean_text(ex["text"]), "label": ex["label"]}
        for ex in ds.select(range(clean_n))
    ]

    combined = adv_pairs_repeated + clean_pairs
    random.shuffle(combined)

    stats = {
        "adv_unique":   len(adv_pairs),
        "adv_repeated": len(adv_pairs_repeated),
        "clean":        len(clean_pairs),
        "total":        len(combined),
    }
    print(f"  {len(adv_pairs)} adv (x{ADV_OVERSAMPLE}) + {len(clean_pairs)} clean"
          f" = {len(combined)} total training samples")

    return encode_dataset_list(combined, tokenizer), stats

def fine_tune(model, train_loader, val_loader,
              epochs: int = DEFENSE_EPOCHS, lr: float = DEFENSE_LR) -> list:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn   = torch.nn.CrossEntropyLoss()
    history   = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"  Defense epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(DEVICE)
            attn_mask = batch["attention_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        # evaluate() from train.py calls model.eval() internally.
        val_acc  = evaluate(model, val_loader)

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        history.append({
            "epoch":        epoch + 1,
            "loss":         round(avg_loss, 6),
            "val_accuracy": round(val_acc, 6),
        })

    return history


def main():
    print("=" * 60)
    print("ADVERSARIAL DEFENSE — Fine-Tuning Pipeline")
    print("=" * 60)

    set_seed()
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)


    print("\n[1/6] Loading tokenizer and model checkpoint...")
    tokenizer      = load_tokenizer_from_checkpoint(CHECKPOINT_IN)
    baseline_model = load_model_from_checkpoint(CHECKPOINT_IN, tokenizer)
    defended_model = load_model_from_checkpoint(CHECKPOINT_IN, tokenizer)

    print("\n[2/6] Loading adversarial examples...")
    with open(ADV_EXAMPLES_PATH) as f:
        adv_examples = json.load(f)
    print(f"  {len(adv_examples)} adversarial examples loaded.")

    if len(adv_examples) == 0:
        print("  No adversarial examples found.")
        print("  Run attack.py first and ensure it produced successful flips.")
        return

    print("\n[3/6] Building clean validation loader (2000 IMDB test samples)...")
    ds_test = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
    val_pairs = [
        {"text": clean_text(ex["text"]), "label": ex["label"]}
        for ex in ds_test.select(range(2000))
    ]
    val_loader = DataLoader(
        encode_dataset_list(val_pairs, tokenizer),
        batch_size=BATCH_SIZE,
    )

    print("\n[4/6] Evaluating baseline model on clean test set...")
    baseline_acc = evaluate(baseline_model, val_loader)
    print(f"  Baseline clean accuracy: {baseline_acc:.4f}")

    print("\n[5/6] Building augmented training dataset...")
    train_encoded, dataset_stats = build_augmented_dataset(adv_examples, tokenizer)
    train_loader = DataLoader(train_encoded, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n[6/6] Fine-tuning for {DEFENSE_EPOCHS} epochs (lr={DEFENSE_LR})...")
    history = fine_tune(defended_model, train_loader, val_loader)

    print("\nEvaluating defended model on clean test set...")
    defended_acc = evaluate(defended_model, val_loader)
    print(f"  Defended clean accuracy: {defended_acc:.4f}")


    ckpt_orig = torch.load(CHECKPOINT_IN, map_location="cpu")
    torch.save(
        {
            "model_state_dict": defended_model.state_dict(),
            "tokenizer_path":   ckpt_orig.get("tokenizer_path", "data/imdb_tokenizer.json"),
            "config":           ckpt_orig["config"],
            "defense_history":  history,
        },
        CHECKPOINT_OUT,
    )
    print(f"\nDefended checkpoint saved -> {CHECKPOINT_OUT}")


if __name__ == "__main__":
    main()
