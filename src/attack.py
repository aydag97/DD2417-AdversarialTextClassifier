
import json
import random
import re
import numpy as np
from collections import defaultdict

import nltk
from nltk.corpus import wordnet

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from load_model import load_model, predict


CANDIDATES_PATH = "outputs/all_high_gradient_candidates.json"
RESULTS_PATH    = "outputs/adversarial_results.json"
EXAMPLES_PATH   = "outputs/adversarial_examples.json"
REPORT_PATH     = "outputs/attack_report.txt"

TOP_K_ATTACK = 10

MIN_REVIEW_CANDIDATES = 3

MAX_SYNONYMS = 5

RANDOM_N = 3
N_REVIEWS = 30

LABEL_TO_INT = {"negative": 0, "positive": 1}

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "to", "of", "in", "on",
    "for", "with", "and", "or", "but", "that", "this", "these", "those",
    "from", "there", "their", "it", "its", "be", "been", "being",
    "at", "by", "as", "if", "so", "do", "did", "does", "has", "have",
    "had", "not", "no", "nor", "yet", "i", "we", "you", "he", "she",
    "they", "me", "him", "her", "us", "my", "your", "his", "our",
    "can", "will", "would", "could", "should", "may", "might", "shall",
    "am", "are",
}

RANDOM_POOL = [
    "table", "window", "garden", "river", "forest",
    "chair", "planet", "machine", "building", "bridge",
]



def is_valid(word: str) -> bool:
    w = word.lower().strip()
    return w not in STOPWORDS and w.isalpha() and len(w) > 2


def get_confidence(probs) -> float:
    return float(np.max(np.asarray(probs)))


def replace_first(text: str, word: str, replacement: str) -> str:
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return pattern.sub(replacement, text, count=1)


def get_synonyms(word: str) -> list:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if (
                candidate != word.lower()
                and " " not in candidate        # single word only
                and candidate.isalpha()
            ):
                synonyms.add(candidate)
    return list(synonyms)


def random_attack(text: str, n: int = RANDOM_N) -> str:
    words   = text.split()
    indices = [i for i, w in enumerate(words) if is_valid(w)]
    chosen  = random.sample(indices, min(n, len(indices)))
    new_words = words[:]
    for i in chosen:
        new_words[i] = random.choice(RANDOM_POOL)
    return " ".join(new_words)


def select_candidates(review_id: int, global_candidates: list, top_k: int = TOP_K_ATTACK) -> list:

    review_specific = sorted(
        [c for c in global_candidates if c["review_id"] == review_id],
        key=lambda x: x["norm_grad"],
        reverse=True,
    )

    if len(review_specific) >= MIN_REVIEW_CANDIDATES:
        return review_specific[:top_k]

  
    review_words = {c["token"].lower() for c in review_specific}
    global_fill  = sorted(
        [c for c in global_candidates if c["token"].lower() not in review_words],
        key=lambda x: x["norm_grad"],
        reverse=True,
    )
    return (review_specific + global_fill)[:top_k]

def attack_review(review: dict, model, tokenizer, global_candidates: list) -> list:
    rid  = review["review_id"]
    text = review["text"]

 
    orig_label, orig_probs = predict(text, model, tokenizer)
    orig_conf              = get_confidence(orig_probs)

    n_review_specific = len([c for c in global_candidates if c["review_id"] == rid])
    used_global_fallback = n_review_specific < MIN_REVIEW_CANDIDATES

    results = []
    candidates = select_candidates(rid, global_candidates)

    for cand in candidates:
        word = cand["token"]
        grad = cand["norm_grad"]

        if not is_valid(word):
            continue

        if not re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE):
            continue

        synonyms = get_synonyms(word)

        if not synonyms:
            results.append({
                "review_id":           rid,
                "strategy":            "high_gradient",
                "word":                word,
                "replacement":         None,
                "flipped":             False,
                "original_label":      orig_label,
                "new_label":           orig_label,
                "gradient":            round(grad, 6),
                "original_conf":       round(orig_conf, 6),
                "new_conf":            round(orig_conf, 6),
                "conf_change":         0.0,
                "attacked_text":       text,
                "note":                "no_synonyms",
                "used_global_fallback": used_global_fallback,
            })
            continue

        for syn in synonyms[:MAX_SYNONYMS]:
            attacked_text = replace_first(text, word, syn)

            if attacked_text == text:
                continue

            new_label, new_probs = predict(attacked_text, model, tokenizer)
            new_conf             = get_confidence(new_probs)
            flipped              = (new_label != orig_label)

            results.append({
                "review_id":           rid,
                "strategy":            "high_gradient",
                "word":                word,
                "replacement":         syn,
                "flipped":             flipped,
                "original_label":      orig_label,
                "new_label":           new_label,
                "gradient":            round(grad, 6),
                "original_conf":       round(orig_conf, 6),
                "new_conf":            round(new_conf, 6),
                "conf_change":         round(new_conf - orig_conf, 6),
                "attacked_text":       attacked_text,
                "note":                "flip" if flipped else "no_flip",
                "used_global_fallback": used_global_fallback,
            })

            if flipped:
                break

    rand_text              = random_attack(text)
    rand_label, rand_probs = predict(rand_text, model, tokenizer)
    rand_conf              = get_confidence(rand_probs)

    results.append({
        "review_id":           rid,
        "strategy":            "random",
        "word":                None,
        "replacement":         None,
        "flipped":             rand_label != orig_label,
        "original_label":      orig_label,
        "new_label":           rand_label,
        "gradient":            None,
        "original_conf":       round(orig_conf, 6),
        "new_conf":            round(rand_conf, 6),
        "conf_change":         round(rand_conf - orig_conf, 6),
        "attacked_text":       rand_text,
        "note":                "random_baseline",
        "used_global_fallback": False,
    })

    return results


def write_report(all_results: list, path: str = REPORT_PATH):
    grad_results   = [r for r in all_results if r["strategy"] == "high_gradient"]
    random_results = [r for r in all_results if r["strategy"] == "random"]

    grad_attempts  = len(grad_results)
    grad_flips     = sum(r["flipped"] for r in grad_results)
    grad_asr       = grad_flips / grad_attempts if grad_attempts else 0
    grad_conf_chg  = [r["conf_change"] for r in grad_results]
    grad_avg_chg   = float(np.mean(grad_conf_chg)) if grad_conf_chg else 0
    flip_chg   = [r["conf_change"] for r in grad_results if r["flipped"]]
    nflip_chg  = [r["conf_change"] for r in grad_results if not r["flipped"]]


    flipped_reviews = len({r["review_id"] for r in grad_results if r["flipped"]})
    total_reviews   = len({r["review_id"] for r in all_results})


    rand_attempts = len(random_results)
    rand_flips    = sum(r["flipped"] for r in random_results)
    rand_asr      = rand_flips / rand_attempts if rand_attempts else 0
    rand_avg_chg  = float(np.mean([r["conf_change"] for r in random_results])) if random_results else 0

    with open(path, "w", encoding="utf-8") as f:

        def w(line=""):
            f.write(line + "\n")

        w("=" * 70)
        w("  ADVERSARIAL ATTACK REPORT")
        w("  Gradient-Guided Synonym Substitution vs Random Baseline")
        w("=" * 70)
        w()

        w("─" * 70)
        w("ATTEMPT LOG  (all attempts, successes and failures)")
        w("─" * 70)

        prev_rid = None
        for r in all_results:
            rid = r["review_id"]
            if rid != prev_rid:
                w()
                w(f"  REVIEW {rid}  |  original_label={r['original_label']}"
                  f"  conf={r['original_conf']:.4f}")
                prev_rid = rid

            if r["strategy"] == "random":
                w(f"    [RANDOM]  flipped={r['flipped']}"
                  f"  new_label={r['new_label']}"
                  f"  conf: {r['original_conf']:.4f} -> {r['new_conf']:.4f}"
                  f"  delta={r['conf_change']:+.4f}")
            else:
                fb   = " [GLOBAL_FALLBACK]" if r.get("used_global_fallback") else ""
                word = r["word"] or "N/A"
                repl = r["replacement"] or "(no synonyms)"
                w(f"    [GRAD{fb}]"
                  f"  '{word}' -> '{repl}'"
                  f"  grad={r['gradient']:.4f}"
                  f"  flipped={r['flipped']}"
                  f"  ({r['note']})"
                  f"  conf: {r['original_conf']:.4f} -> {r['new_conf']:.4f}"
                  f"  delta={r['conf_change']:+.4f}")
        w()

        w("─" * 70)
        w("PER-REVIEW SUMMARY  (first successful flip per review)")
        w("─" * 70)
        w()

        by_review = defaultdict(list)
        for r in grad_results:
            by_review[r["review_id"]].append(r)

        for rid in sorted(by_review.keys()):
            attempts = by_review[rid]
            flips    = [a for a in attempts if a["flipped"]]
            if flips:
                first = flips[0]
                w(f"  Review {rid:>3}:  FLIPPED"
                  f"  '{first['word']}' -> '{first['replacement']}'"
                  f"  grad={first['gradient']:.4f}"
                  f"  delta={first['conf_change']:+.4f}")
            else:
                w(f"  Review {rid:>3}:  no flip  ({len(attempts)} attempts)")
        w()

        w("=" * 70)
        w("AGGREGATE STATISTICS")
        w("=" * 70)
        w()
        w("  GRADIENT-GUIDED ATTACK")
        w(f"    Total attempts          : {grad_attempts}")
        w(f"    Successful flips        : {grad_flips}")
        w(f"    Attack Success Rate     : {grad_asr:.4f}  ({grad_asr*100:.1f}%)")
        w(f"    Reviews with >= 1 flip  : {flipped_reviews} / {total_reviews}")
        w(f"    Avg confidence change   : {grad_avg_chg:+.4f}")
        if flip_chg:
            w(f"    Avg delta (flipped)     : {float(np.mean(flip_chg)):+.4f}")
        if nflip_chg:
            w(f"    Avg delta (not flipped) : {float(np.mean(nflip_chg)):+.4f}")
        w()
        w("  RANDOM BASELINE")
        w(f"    Total attempts          : {rand_attempts}")
        w(f"    Successful flips        : {rand_flips}")
        w(f"    Attack Success Rate     : {rand_asr:.4f}  ({rand_asr*100:.1f}%)")
        w(f"    Avg confidence change   : {rand_avg_chg:+.4f}")
        w()
        w("  COMPARISON")
        if rand_asr > 0:
            lift = (grad_asr - rand_asr) / rand_asr * 100
            w(f"    Gradient ASR lift over random: {lift:+.1f}%")
        else:
            w("    Gradient ASR lift over random: N/A (random ASR = 0)")
        w()
        w("=" * 70)

    print(f"Report saved -> {path}")


def main():
    from datasets import load_dataset as _load_dataset

    print("=" * 60)
    print("ADVERSARIAL ATTACK")
    print("=" * 60)

    model, tokenizer = load_model()

    # Load gradient candidates produced by visualise_mod.py.
    with open(CANDIDATES_PATH) as f:
        global_candidates = json.load(f)
    print(f"Loaded {len(global_candidates)} gradient candidates.\n")

    ds = _load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
    reviews = []
    for i, ex in enumerate(ds):
        text = re.sub(r"<br\s*/?>", " ", ex["text"])
        text = re.sub(r"\s+", " ", text).strip()
        reviews.append({"review_id": i, "text": text})
        if i >= N_REVIEWS - 1:   # 0-indexed: N_REVIEWS=30 -> ids 0..29
            break

    print(f"Attacking {len(reviews)} reviews...\n")

    all_results = []
    for rev in reviews:
        results = attack_review(rev, model, tokenizer, global_candidates)
        all_results.extend(results)

        grad_attempts = [r for r in results if r["strategy"] == "high_gradient"]
        flips         = sum(r["flipped"] for r in grad_attempts)
        print(
            f"  Review {rev['review_id']:>3}:  "
            f"{len(grad_attempts)} attempts,  {flips} flip(s)"
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results -> {RESULTS_PATH}")


    adv_examples = []
    seen_texts   = set()

    for r in all_results:
        if r["strategy"] == "high_gradient" and r["flipped"]:
            t = r["attacked_text"]
            if t in seen_texts:
                continue
            # Convert string label to int.  If predict() returns a label
            # not in LABEL_TO_INT, we skip with a warning rather than
            # silently assigning the wrong class.
            int_label = LABEL_TO_INT.get(r["original_label"])
            if int_label is None:
                print(f"  WARNING: unknown label '{r['original_label']}' "
                      f"for review {r['review_id']} — skipping.")
                continue
            adv_examples.append({
                "text":               t,
                "label":              int_label,
                "original_label_str": r["original_label"],
                "word_replaced":      r["word"],
                "replacement":        r["replacement"],
                "review_id":          r["review_id"],
            })
            seen_texts.add(t)

    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(adv_examples, f, indent=2)
    print(f"Adversarial examples -> {EXAMPLES_PATH}  ({len(adv_examples)} samples)")

    write_report(all_results)

    print("\nNext step: python defense.py")


if __name__ == "__main__":
    main()
