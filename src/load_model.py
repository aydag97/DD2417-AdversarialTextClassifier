import torch
from model import TransformerClassifier
from tokenizer import TinyStoriesTokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pad_sequence(ids, block_size):
    ids = ids[:block_size]
    attention_mask = [1] * len(ids)

    while len(ids) < block_size:
        ids.append(0)
        attention_mask.append(0)

    return ids, attention_mask


def load_model(path="checkpoints/baseline_2layer_512.pt"):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = TransformerClassifier(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    tokenizer = TinyStoriesTokenizer.load(checkpoint["tokenizer_path"])

    return model, tokenizer


def predict(text, model, tokenizer):
    block_size = 256

    _, ids = tokenizer.tokenize(text)
    ids, attention_mask = pad_sequence(ids, block_size)

    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probs, dim=1).item()

    label = "positive" if prediction == 1 else "negative"

    return label, probs.cpu().numpy()




def get_gradients(text, model, tokenizer):
    block_size = 256

    _, ids = tokenizer.tokenize(text)
    ids, attention_mask = pad_sequence(ids, block_size)

    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(DEVICE)

    embeddings = model.token_embedding(input_ids)
    embeddings.retain_grad()

    positions = torch.arange(input_ids.shape[1], device=DEVICE)
    positions = positions.unsqueeze(0)

    x = embeddings + model.position_embedding(positions)

    x = model.blocks(x)

    mask = attention_mask.unsqueeze(-1)
    x = x * mask

    pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    logits = model.classifier(pooled)

    prediction = torch.argmax(logits, dim=1)

    selected_logit = logits[0, prediction]

    model.zero_grad()
    selected_logit.backward()

    gradients = embeddings.grad.norm(dim=-1).squeeze(0)

    return gradients.cpu().numpy()


if __name__ == "__main__":
    model, tokenizer = load_model()

if __name__ == "__main__":
    model, tokenizer = load_model()

    examples = [
        "This movie was amazing and beautiful.",
        "This movie was terrible and boring.",
        "I loved this film.",
        "I hated this film.",
        "The acting was excellent.",
        "The acting was awful.",
    ]

    for text in examples:
        label, probs = predict(text, model, tokenizer)
        print(text)
        print(label, probs)
        print()

    # grads = get_gradients(
    #     "This movie was amazing and beautiful.",
    #     model,
    #     tokenizer,
    # )

    # print(grads[:20])