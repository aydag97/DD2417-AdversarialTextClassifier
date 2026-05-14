import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import random
import numpy as np
import os

from tokenizer import TinyStoriesTokenizer
from model import TransformerClassifier


BLOCK_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_sequence(ids, block_size):
    ids = ids[:block_size]

    attention_mask = [1] * len(ids)

    while len(ids) < block_size:
        ids.append(0)
        attention_mask.append(0)

    return ids, attention_mask



def build_tokenizer():

    tokenizer_path = "data/imdb_tokenizer.json"

    if os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        return TinyStoriesTokenizer.load(tokenizer_path)

    print("Training tokenizer...")

    dataset = load_dataset("imdb")["train"]

    with open("data/imdb_text.txt", "w", encoding="utf-8") as f:
        for text in dataset["text"]:
            f.write(text + "\n")

    tokenizer = TinyStoriesTokenizer(vocab_size=5000)

    tokenizer.train("data/imdb_text.txt")

    tokenizer.save(tokenizer_path)

    return tokenizer


def encode_dataset(dataset, tokenizer):
    encoded = []

    for text, label in zip(dataset["text"], dataset["label"]):
        _, ids = tokenizer.tokenize(text)

        ids, attention_mask = pad_sequence(ids, BLOCK_SIZE)

        encoded.append(
            {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.tensor(attention_mask),
                "label": torch.tensor(label),
            }
        )

    return encoded


def evaluate(model, dataloader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask)

            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def train():
    set_seed()

    full_train = load_dataset("imdb")["train"].shuffle(seed=42)
    full_test = load_dataset("imdb")["test"].shuffle(seed=42)

    train_dataset = full_train.select(range(25000))
    test_dataset = full_test.select(range(5000))

    tokenizer = build_tokenizer()

    train_encoded = encode_dataset(train_dataset, tokenizer)
    test_encoded = encode_dataset(test_dataset, tokenizer)

    train_loader = DataLoader(
        train_encoded,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_encoded,
        batch_size=BATCH_SIZE,
    )

    model = TransformerClassifier(
        vocab_size=len(tokenizer.vocab),
        block_size=BLOCK_SIZE,
        vector_dim=128,
        n_heads=4,
        n_layers=2,
        hidden_dim=256,
        dropout=0.2
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    loss_fn = torch.nn.CrossEntropyLoss()
    best_accuracy = 0
    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)

            loss = loss_fn(logits, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        accuracy = evaluate(model, test_loader)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_path": "data/imdb_tokenizer.json",
                    "config": {
                        "vocab_size": len(tokenizer.vocab),
                        "block_size": BLOCK_SIZE,
                        "vector_dim": 128,
                        "n_heads": 4,
                        "n_layers": 2,
                        "hidden_dim": 256,
                    },
                },
                "checkpoints/baseline_2layer_512.pt",
            )

    print("Best model saved.")


if __name__ == "__main__":
    train()