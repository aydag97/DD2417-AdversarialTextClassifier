
import os
import re
import json
import numpy as np
from datasets import load_dataset

from load_model import load_model, predict, get_gradients


try:
    from gradient_explorer import GradientExplorer
    _EXPLORER = GradientExplorer(aggregation="max")
    EXPLORER_AVAILABLE = True
except ImportError:
    _EXPLORER = None
    EXPLORER_AVAILABLE = False


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


TOP_K_PER_REVIEW = 10
N_REVIEWS = 200


# Filter out stopwords, punctuation, and very short tokens
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


def safe_confidence(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim > 1:
        probs = probs.reshape(-1)

    return float(np.max(probs))

# Merge tokens, the final word receives the maximum gradient of its sub-tokens
def merge_tokens_to_words(tokens, grads):
    words, word_grads = [], []

    current_word = ""
    current_grads = []

    for t, g in zip(tokens, grads):

        if t.startswith("##"):
            current_word += t[2:]
            current_grads.append(g)

        else:
            if current_word:
                words.append(current_word)
                word_grads.append(float(max(current_grads)))

            current_word = t
            current_grads = [g]

    if current_word:
        words.append(current_word)
        word_grads.append(float(max(current_grads)))

    return words, word_grads

# Run sentiment prediction and gradient extraction for a review,
# then produce a ranked list of words ordered by importance.
def analyze_review(text: str, model, tokenizer):

    label, probs = predict(text, model, tokenizer)

    raw_grads = get_gradients(text, model, tokenizer)
    tokens, _ = tokenizer.tokenize(text)

    n = min(len(tokens), len(raw_grads))
    tokens = tokens[:n]
    raw_grads = raw_grads[:n]

    grad_scores = raw_grads

    if EXPLORER_AVAILABLE:
        word_scores = _EXPLORER.analyse(text, grad_scores, tokenizer, top_k=0)
        words = [w.word for w in word_scores]
        word_grads = [w.score for w in word_scores]
    else:
        words, word_grads = merge_tokens_to_words(tokens, grad_scores.tolist())

    rows = []
    for w, g in zip(words, word_grads):
        if not is_valid_word(w):
            continue

        rows.append({
            "token": w,
            "norm_grad": float(g),
            "prediction": label,
            "probs": probs.tolist() if hasattr(probs, "tolist") else probs,
            "review_text": text
        })

    rows.sort(key=lambda x: x["norm_grad"], reverse=True)
    return rows, label, probs


def load_reviews(n=N_REVIEWS):
    ds = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
    out = []
    for i, ex in enumerate(ds):
        text = re.sub(r"<br\s*/?>", " ", ex["text"])
        text = re.sub(r"\s+", " ", text).strip()

        out.append({
            "review_id": i,
            "text": text,
            "label": ex["label"]
        })

        if len(out) >= n:
            break
    return out

def main():

    print("=" * 60)
    print("GRADIENT EXTRACTION PIPELINE")
    print("=" * 60)
    model, tokenizer = load_model()
    reviews = load_reviews(N_REVIEWS)
    print(f"Loaded {len(reviews)} reviews\n")

    os.makedirs("outputs", exist_ok=True)

    global_candidates = []
    full_report = []

    for r in reviews:

        rows, label, probs = analyze_review(r["text"], model, tokenizer)

        conf = safe_confidence(probs)

        top = rows[:TOP_K_PER_REVIEW]

        for item in top:
            global_candidates.append({
                "review_id": r["review_id"],
                "token": item["token"],
                "norm_grad": item["norm_grad"],
                "label": label
            })

        full_report.append({
            "review_id": r["review_id"],
            "text": r["text"],
            "predicted_label": label,
            "confidence": conf,
            "top_words": rows[:50]
        })

        print(f"Review {r['review_id']} | conf={conf:.3f} | top={top[0]['token'] if top else 'N/A'}")

    with open("outputs/all_high_gradient_candidates.json", "w") as f:
        json.dump(global_candidates, f, indent=2)

    with open("outputs/gradient_word_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\nSaved:")
    print("✔ outputs/all_high_gradient_candidates.json")
    print("✔ outputs/gradient_word_report.json")


if __name__ == "__main__":
    main()
