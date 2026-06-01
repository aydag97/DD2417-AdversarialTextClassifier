import os
import csv
import html 
import re   #to clean IMDb text
import numpy as np
import matplotlib.pyplot as plt
from load_model import load_model, predict, get_gradients 
from datasets import load_dataset #IMDb test reviews
import nltk #sentiment lexicon
from nltk.corpus import opinion_lexicon

nltk.download("opinion_lexicon") #opinion_lexicon contains sentiment words 

#Spacy is used for POS tagging
#it helps comparing whether adjectives, nouns, verbs, or adverbs receive higher gradient score
#ADJ = adjective
#NOUN = noun
#VERB = verb
#ADV = adverb
try:
    import spacy
    NLP = spacy.load("en_core_web_sm") #loads a pre-trained English spaCy model
    SPACY_AVAILABLE = True
except Exception:
    NLP = None
    SPACY_AVAILABLE = False

#normalize gradient scores to the range [0, 1]
#easier to plot
def normalize_scores(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    if scores.max() == scores.min(): #are all values the same?
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def clean_token(token):
    return token.strip() #remove whitespace from token


def is_meaningful_token(token):
    token = clean_token(token)

    if token == "": #if token is empty
        return False
    if all(char in ".,!?;:-()[]{}\"'" for char in token): #if the token is only punctuation
        return False
    
    return True

def is_punctuation_token(token):
    token = clean_token(token)

    if token == "":
        return True
    
    return all(char in ".,!?;:-()[]{}\"'" for char in token)


def is_readable_token(row):
    #keep tokens that are useful and not subword tokens
    token = clean_token(row["token"])

    if token == "":
        return False
    
    if not row["meaningful_token"]:
        return False
    
    if row["pos"] == "PUNCT":
        return False
    
    if row["pos"] == "UNKNOWN":
        return False
    
    #remove short subword tokens that don't make sense
    if len(token) <= 1:
        return False
    
    return True


def merge_subword_tokens(rows):
    """
    Merge subword tokens into readable words for visualization

    Example:
    The tokenizer can split words like "frustrating" into
    "fr", "ustr", "ating" 
    
    So we merge it back into "frustrating"

    The gradient score of the merged word is the maximum score
    among its subword pieces
    """

    merged_rows = []
    current_token = ""
    current_rows = [] #all subword rows belonging to the current word

    for row in rows:
        token = row["token"]

        #If token starts with a space, it means a new word starts
        if token.startswith(" ") and current_rows:
            merged_rows.append(create_merged_row(current_token, current_rows))

            current_token = token
            current_rows = [row]

        else:
            current_token += token
            current_rows.append(row)

    #add last word
    if current_rows:
        merged_rows.append(create_merged_row(current_token, current_rows))

    return merged_rows


def create_merged_row(token, rows):
    cleaned = clean_token(token)

    #Ignore punctuation when calculating the displayed word score
    non_punctuation_rows = [
        row for row in rows
        if not is_punctuation_token(row["token"])
    ]

    #If everything was punctuation, keep the original rows as fallback
    if not non_punctuation_rows:
        non_punctuation_rows = rows


    max_row = max(
        non_punctuation_rows,
        key=lambda row: row["normalized_gradient_score"]
    )

    return {
        "token": cleaned.strip(".,!?;:-()[]{}\"'"),
        "clean_token": cleaned.strip(".,!?;:-()[]{}\"'"),
        "normalized_gradient_score": max(
            row["normalized_gradient_score"] for row in non_punctuation_rows
        ),
        "raw_gradient_score": max(
            row["raw_gradient_score"] for row in non_punctuation_rows
        ),
        "pos": max_row["pos"],
        "sentiment_category": max_row["sentiment_category"],
        "meaningful_token": is_meaningful_token(cleaned),
        "prediction": max_row["prediction"],
        "negative_probability": max_row["negative_probability"],
        "positive_probability": max_row["positive_probability"],
    }

#finds sentiment words from the IMDb training data using NLTK’s opinion lexicon
def get_dataset_sentiment_words(dataset, text_column="text"):
    positive_lexicon = set(opinion_lexicon.positive())
    negative_lexicon = set(opinion_lexicon.negative())

    dataset_positive_words = set()
    dataset_negative_words = set()

    for example in dataset:
        text = example[text_column]
        tokens = text.split()

        for token in tokens:
            token = clean_token(token).lower()

            if not is_meaningful_token(token):
                continue

            if token in positive_lexicon:
                dataset_positive_words.add(token)

            if token in negative_lexicon:
                dataset_negative_words.add(token)

    return dataset_positive_words, dataset_negative_words


#One required experiment is to analyze whether sentiment-heavy words receive larger gradients
#this checks if a token is a sentiment word
def simple_sentiment_category(token, positive_words, negative_words):
    token = clean_token(token).lower()

    if token in positive_words:
        return "positive_sentiment"
    if token in negative_words:
        return "negative_sentiment"
    
    return "other"

#give each token a pos tag
def get_pos_tags_for_text(text, tokens):
    if not SPACY_AVAILABLE:
        return ["UNKNOWN"] * len(tokens)
    
    doc = NLP(text)

    word_to_pos = {}
    for word in doc:
        word_to_pos[word.text.lower()] = word.pos_

    pos_tags = []
    for token in tokens:
        cleaned = clean_token(token).lower()
        pos_tags.append(word_to_pos.get(cleaned, "UNKNOWN"))

    return pos_tags



def analyze_text(text, model, tokenizer, positive_words, negative_words):
    """
    1. predict positive or negative
    2. then tokenize the text
    3. compute gradient importance scores
    4. lastly, match tokens with their gradient scores
    """  
  
    label, probs = predict(text, model, tokenizer)

    #tokenizer returns 2 things:
    #tokens = readable token strings
    #ids = token IDs used by the model
    tokens, ids = tokenizer.tokenize(text)
    gradients = get_gradients(text, model, tokenizer)

    #get_gradients returns at most 256 scores because load_model.py uses block_size = 256.
    #IMDb reviews can have more than 256 tokens, so we only analyze tokens that have gradient scores.
    usable_length = min(len(tokens), len(gradients))

    tokens = tokens[:usable_length]
    raw_scores = gradients[:usable_length]

    #Normalize so plots are easier to compare
    normalized_scores = normalize_scores(raw_scores)

    pos_tags = get_pos_tags_for_text(text, tokens)

    rows = []

    for index, token in enumerate(tokens):
        cleaned = clean_token(token)

        rows.append({
            "text": text,
            "prediction": label,
            "negative_probability": float(probs[0][0]),
            "positive_probability": float(probs[0][1]),
            "token_index": index,
            "token": token,
            "clean_token": cleaned,
            "raw_gradient_score": float(raw_scores[index]),
            "normalized_gradient_score": float(normalized_scores[index]),
            "pos": pos_tags[index],
            "sentiment_category": simple_sentiment_category(
                cleaned,
                positive_words,
                negative_words
            ),
            "meaningful_token": is_meaningful_token(cleaned),
        })

    return rows, label, probs



def print_token_importance(text, rows, label, probs):
    #print tokens sorted from most important to least important

    rows = merge_subword_tokens(rows)

    print("=" * 80)
    print("TEXT:")
    print(text)
    print()
    print("PREDICTION:", label)
    print("PROBABILITIES:", probs)
    print()

    readable_rows = [row for row in rows if is_readable_token(row)]

    ranked = sorted(
        readable_rows, 
        key=lambda row: row["normalized_gradient_score"],
        reverse=True,
    )

    print("MOST IMPORTANT TOKENS:")
    for row in ranked:
        print(
            f"{repr(row['token']):20s} "
            f"{row['normalized_gradient_score']:.4f} "
            f"POS={row['pos']:8s} "
            f"{row['sentiment_category']}"
        )



def plot_token_importance(rows, title, output_path, top_k=20):
    """
    Plot only the top-k most important meaningful tokens.    
    """
    rows = merge_subword_tokens(rows)

    filtered_rows = [row for row in rows if is_readable_token(row)]

    #sort by importance, highest first
    ranked = sorted(
        filtered_rows,
        key=lambda row: row["normalized_gradient_score"],
        reverse=True,
    )

    #keep only top-k
    ranked = ranked[:top_k]

    #reverse so the biggest bar appears at the top nicely in barh
    ranked = ranked[::-1]

    tokens = [row["clean_token"] for row in ranked]
    scores = [row["normalized_gradient_score"] for row in ranked]

    plt.figure(figsize=(10, 8))
    plt.barh(tokens, scores)
    plt.xlabel("Normalized gradient importance")
    plt.title(f"{title} (Top {top_k} tokens)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_colored_html(rows, output_path, title):
    """
    Create an HTML heatmap.
    Darker color means higher gradient importance.
    """
    rows = merge_subword_tokens(rows)

    html_tokens = []

    for row in rows:
        display_token = row["clean_token"].strip(".,!?;:-()[]{}\"'")

        if display_token == "":
            continue

        token = html.escape(display_token)
        score = row["normalized_gradient_score"]

        intensity = int(255 - score * 180)

        token_html = (
            f'<span title="score={score:.4f}, POS={row["pos"]}" '
            f'style="background-color: rgb({intensity},{intensity},255); '
            f'padding: 3px; margin: 2px; display: inline-block;">'
            f'{token}'
            f'</span>'
        )

        html_tokens.append(token_html)

    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 2;">
        <h2>{title}</h2>
        <p>{" ".join(html_tokens)}</p>
        <p><b>Darker blue = higher gradient importance</b></p>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_content)


def save_rows_to_csv(rows, output_path):
    """
    Save token-level results to CSV
    """
    fieldnames = [
        "token_index",
        "token",
        "clean_token",
        "raw_gradient_score",
        "normalized_gradient_score",
        "pos",
        "sentiment_category",
        "meaningful_token",
        "prediction",
        "negative_probability",
        "positive_probability",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def save_summary_by_label(all_rows, output_path):
    """
    Compare average gradient scores for positive vs negative predictions.
    """
    labels = sorted(set(row["prediction"] for row in all_rows))

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "prediction",
            "num_tokens",
            "avg_normalized_gradient",
            "avg_raw_gradient",
        ])

        for label in labels:
            rows = [
                row for row in all_rows
                if row["prediction"] == label and is_readable_token(row)
            ]

            if not rows:
                continue

            avg_norm = np.mean([row["normalized_gradient_score"] for row in rows])
            avg_raw = np.mean([row["raw_gradient_score"] for row in rows])

            writer.writerow([
                label,
                len(rows),
                avg_norm,
                avg_raw,
            ])


def save_summary_by_pos(all_rows, output_path):
    """
    Compare gradient importance across POS tags.
    """
    pos_tags = sorted(set(row["pos"] for row in all_rows))

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "pos",
            "num_tokens",
            "avg_normalized_gradient",
            "avg_raw_gradient",
        ])

        for pos in pos_tags:
            rows = [
                row for row in all_rows
                if row["pos"] == pos and is_readable_token(row)
            ]

            if not rows:
                continue

            avg_norm = np.mean([row["normalized_gradient_score"] for row in rows])
            avg_raw = np.mean([row["raw_gradient_score"] for row in rows])

            writer.writerow([
                pos,
                len(rows),
                avg_norm,
                avg_raw,
            ])

#Do sentiment-heavy words receive larger gradients?
def save_summary_by_sentiment_category(all_rows, output_path):
    """
    Compare sentiment-heavy words against other words.
    """
    categories = sorted(set(row["sentiment_category"] for row in all_rows))

    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sentiment_category",
            "num_tokens",
            "avg_normalized_gradient",
            "avg_raw_gradient",
        ])

        for category in categories:
            rows = [
                row for row in all_rows
                if row["sentiment_category"] == category and is_readable_token(row)
            ]

            if not rows:
                continue

            avg_norm = np.mean([row["normalized_gradient_score"] for row in rows])
            avg_raw = np.mean([row["raw_gradient_score"] for row in rows])

            writer.writerow([
                category,
                len(rows),
                avg_norm,
                avg_raw,
            ])


def load_imdb_examples(num_positive=3, num_negative=3, max_chars=1000):
    """
    Load IMDb reviews from the test set.

    IMDb labels:
    0 = negative
    1 = positive
    """

    dataset = load_dataset("stanfordnlp/imdb")["test"].shuffle(seed=42)
    
    examples = []
    positive_count = 0
    negative_count = 0

    for item in dataset:
        text = item["text"]
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        true_label = item["label"]

        #Keep reviews shorter so plots are readable
        text = text[:max_chars]

        if true_label == 1 and positive_count < num_positive:
            examples.append({
                "name": f"imdb_positive_{positive_count + 1}",
                "text": text,
                "true_label": "positive",
            })
            positive_count += 1

        elif true_label == 0 and negative_count < num_negative:
            examples.append({
                "name": f"imdb_negative_{negative_count + 1}",
                "text": text,
                "true_label": "negative",
            })
            negative_count += 1

        if positive_count == num_positive and negative_count == num_negative:
            break

    return examples


def main():
    model, tokenizer = load_model()

    dataset = load_dataset("stanfordnlp/imdb")
    train_dataset = dataset["train"]

    positive_words, negative_words = get_dataset_sentiment_words(train_dataset)

    print("Positive sentiment words found:", len(positive_words))
    print("Negative sentiment words found:", len(negative_words))

    output_dir = "outputs/kim_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    examples = load_imdb_examples(num_positive=3, num_negative=3)

    all_rows = []

    if not SPACY_AVAILABLE:
        print("SpaCy is not available. POS tags will be UNKNOWN.")
        print("To enable POS analysis, install SpaCy and the English model:")
        print("pip install spacy")
        print("python3 -m spacy download en_core_web_sm")
        print()

    for example in examples:
        name = example["name"]
        text = example["text"]

        rows, label, probs = analyze_text(
            text, 
            model, 
            tokenizer, 
            positive_words, 
            negative_words
        )

        all_rows.extend(rows)

        print_token_importance(text, rows, label, probs)

        plot_path = f"{output_dir}/{name}_barplot.png"
        html_path = f"{output_dir}/{name}_heatmap.html"
        csv_path = f"{output_dir}/{name}_tokens.csv"

        plot_token_importance(
            rows,
            title=f"Gradient Token Importance: {label}",
            output_path=plot_path,
        )

        create_colored_html(
            rows,
            output_path=html_path,
            title=f"Gradient Heatmap: {label}",
        )

        save_rows_to_csv(rows, csv_path)

        print()
        print(f"Saved bar plot to: {plot_path}")
        print(f"Saved HTML heatmap to: {html_path}")
        print(f"Saved token CSV to: {csv_path}")
        print()

    save_rows_to_csv(all_rows, f"{output_dir}/all_token_results.csv")
    save_summary_by_label(all_rows, f"{output_dir}/summary_by_label.csv")
    save_summary_by_pos(all_rows, f"{output_dir}/summary_by_pos.csv")
    save_summary_by_sentiment_category(
        all_rows,
        f"{output_dir}/summary_by_sentiment_category.csv",
    )

    print("=" * 80)
    print("DONE")
    print(f"Saved all results to: {output_dir}")
    print("Main files for report:")
    print(f"- {output_dir}/all_token_results.csv")
    print(f"- {output_dir}/summary_by_label.csv")
    print(f"- {output_dir}/summary_by_pos.csv")
    print(f"- {output_dir}/summary_by_sentiment_category.csv")


if __name__ == "__main__":
    main()