import os
import numpy as np
import matplotlib.pyplot as plt

from load_model import load_model, predict, get_gradients

#normalize gradient scores to the range [0, 1]
def normalize_scores(scores):
    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    if scores.max() == scores.min(): #are all values are the same?
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def analyze_text(text, model, tokenizer):
    """
    For one text:
    1. predict positive or negative
    2. then tokenize the text
    3. compute gradient importance scores
    4. lastly, align tokens with their gradient scores
    """  
  
    label, probs = predict(text, model, tokenizer)

    #tokenizer returns 2 things:
    #tokens = readable token strings
    #ids = token IDs used by the model
    tokens, ids = tokenizer.tokenize(text)

    gradients = get_gradients(text, model, tokenizer)

    #get_gradients returns scores for padded input
    #but we want the real tokens, not padding
    scores = gradients[:len(tokens)]

    #Normalize so plots are easier to compare
    scores = normalize_scores(scores)

    return tokens, scores, label, probs

def print_token_importance(text, tokens, scores, label, probs):
    #print tokens sorted from most important to least important

    print("=" * 80)
    print("TEXT:")
    print(text)
    print()
    print("PREDICTION:", label)
    print("PROBABILITIES:", probs)
    print()

    ranked = sorted(
        zip(tokens, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    print("MOST IMPORTANT TOKENS:")
    for token, score in ranked:
        print(f"{repr(token):20s} {score:.4f}")

def plot_token_importance(tokens, scores, title, output_path):
    #create a bar plot where each token has one importance score

    plt.figure(figsize=(12, 5))
    plt.bar(tokens, scores)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Normalized gradient importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_colored_html(tokens, scores, output_path, title):
    """
    HTML visualization
    more important tokens get stronger background color
    """

    html_tokens = []

    for token, score in zip(tokens, scores):
        intensity = int(255 - score * 180)

        token_html = (
            f'<span style="background-color: rgb(255, {intensity},{intensity}); '
            f'padding: 3px; margin: 2px; display: inline-block;">'
            f'{token}'
            f'</span>'
        )

        html_tokens.append(token_html)

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 2;">
        <h2>{title}</h2>
        <p>{" ".join(html_tokens)}</p>
        <p><b>Darker red = higher gradient importance</b></p>
    </body>
    </html>

    """

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html)

def main():
    model, tokenizer = load_model()

    os.makedirs("outputs/kim_visualizations", exist_ok=True)

    examples = [
        {
            "name": "positive_review_1",
            "text": "This movie was amazing and beautifully acted.",
        },
        {
            "name": "negative_review_1",
            "text": "This movie was terrible, boring, and badly written.",
        },
        {
            "name": "mixed_review_1",
            "text": "The acting was excellent, but the story was slow and predictable.",
        },
        {
            "name": "positive_review_2",
            "text": "I loved this film because it was emotional, funny, and inspiring.",
        },
        {
            "name": "negative_review_2",
            "text": "I hated this film because it was dull, confusing, and too long.",
        },
    ]

    for example in examples:
        name = example["name"]
        text = example["text"]

        tokens, scores, label, probs = analyze_text(text, model, tokenizer)

        print_token_importance(text, tokens, scores, label, probs)

        plot_path = f"outputs/kim_visualizations/{name}_barplot.png"
        html_path = f"outputs/kim_visualizations/{name}_heatmap.html"

        plot_token_importance(
            tokens,
            scores,
            title=f"Gradient Token Importance: {label}",
            output_path=plot_path,
        )

        create_colored_html(
            tokens,
            scores,
            output_path=html_path,
            title=f"Gradient Heatmap: {label}",
        )

        print()
        print(f"Saved bar plot to: {plot_path}")
        print(f"Saved HTML heatmap to: {html_path}")
        print()

if __name__ == "__main__":
    main()

