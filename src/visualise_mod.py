import os
import re
import json
import numpy as np
from datasets import load_dataset

from load_model import load_model, predict, get_gradients

# Optional: richer word-level alignment via surface-text reconstruction.
try:
    from gradient_explorer import GradientExplorer
    _EXPLORER = GradientExplorer(aggregation="max")
    EXPLORER_AVAILABLE = True
except ImportError:
    _EXPLORER = None
    EXPLORER_AVAILABLE = False

# Words whose gradient we do NOT care about even if numerically large.
STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "but", "that", "this", "these", "those",
    "from", "there", "their", "it", "its", "be", "been", "being",
    "at", "by", "as", "if", "so", "do", "did", "does", "has", "have",
    "had", "not", "no", "nor", "yet", "both", "either", "neither",
    "i", "we", "you", "he", "she", "they", "me", "him", "her", "us",
    "my", "your", "his", "our", "can", "will", "would", "could",
    "should", "may", "might", "shall", "am", "are",
}

# How many top-gradient words to keep per review in the candidates file.
TOP_K_PER_REVIEW = 10

N_REVIEWS = 30

def is_valid_word(word: str) -> bool:
    w = word.lower().strip()
    return (
        w != ""
        and w not in STOPWORDS
        and w.isalpha()
        and len(w) > 2
    )


def normalize_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def merge_tokens_to_words(tokens: list, grads: list):
    words, word_grads = [], []
    current_word, current_grads = "", []

    for tok, grad in zip(tokens, grads):
        if tok.startswith("##"):
            current_word += tok[2:]
            current_grads.append(grad)
        else:
            if current_word:
                words.append(current_word)
                word_grads.append(float(max(current_grads)))
            current_word = tok
            current_grads = [grad]

    if current_word:
        words.append(current_word)
        word_grads.append(float(max(current_grads)))

    return words, word_grads


def analyze_review(text: str, model, tokenizer):
    label, probs = predict(text, model, tokenizer)
    raw_grads = get_gradients(text, model, tokenizer)
    tokens, _ = tokenizer.tokenize(text).
    n = min(len(tokens), len(raw_grads))
    tokens    = tokens[:n]
    raw_grads = raw_grads[:n]
    norm_grads = normalize_array(raw_grads)
    if EXPLORER_AVAILABLE:
        word_scores = _EXPLORER.analyse(text, norm_grads, tokenizer, top_k=0)
        words      = [ws.word for ws in word_scores]
        word_grads = [ws.score for ws in word_scores]
    else:
        words, word_grads = merge_tokens_to_words(tokens, norm_grads.tolist())

    rows = []
    for word, grad in zip(words, word_grads):
        if not is_valid_word(word):
            continue
        rows.append({
            "token":     word,
            "norm_grad": round(float(grad), 6),
        })

    rows.sort(key=lambda r: r["norm_grad"], reverse=True)
    return rows, label, probs

def load_reviews(n: int = N_REVIEWS):
    ds = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
    out = []
    for i, ex in enumerate(ds):
        text = re.sub(r"<br\s*/?>", " ", ex["text"])
        text = re.sub(r"\s+", " ", text).strip()
        out.append({
            "review_id": i,
            "text":      text,
            "label":     ex["label"],   # ground-truth: 0=negative, 1=positive
        })
        if len(out) >= n:
            break
    return out

def main():
    print("=" * 60)
    print("GRADIENT EXTRACTION PIPELINE")
    print("=" * 60)

    model, tokenizer = load_model()
    print(f"Model loaded.  GradientExplorer available: {EXPLORER_AVAILABLE}\n")

    reviews = load_reviews(N_REVIEWS)
    print(f"Loaded {len(reviews)} reviews from IMDB test set.\n")

    os.makedirs("outputs", exist_ok=True)

    global_candidates = []
    full_report = []

    for review in reviews:
        rid  = review["review_id"]
        text = review["text"]

        rows, label, probs = analyze_review(text, model, tokenizer)
        conf = float(np.max(np.asarray(probs)))
        top  = rows[:TOP_K_PER_REVIEW]

        for r in top:
            global_candidates.append({
                "review_id": rid,
                "token":     r["token"],
                "norm_grad": r["norm_grad"],
                "label":     label,
            })

        full_report.append({
            "review_id":       rid,
            "text":            text,
            "ground_truth":    review["label"],
            "predicted_label": label,
            "confidence":      round(conf, 6),
            "top_words": [
                {"rank": i + 1, "word": r["token"], "norm_grad": r["norm_grad"]}
                for i, r in enumerate(rows[:50])
            ],
        })

        print(
            f"  [{rid+1:>2}/{N_REVIEWS}]  pred={label}  conf={conf:.3f}  "
            f"top='{top[0]['token'] if top else 'N/A'}'  "
            f"grad={top[0]['norm_grad'] if top else 0:.4f}"
        )


if __name__ == "__main__":
    main()
