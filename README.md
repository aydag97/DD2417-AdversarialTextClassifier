# Adversarial Text Classifier

## Overview

This project investigates adversarial robustness in Transformer-based sentiment classification. Starting from a custom Transformer implementation developed in Assignment 3, we built a non-causal sentiment classifier trained on the IMDb movie review dataset. The project further explores gradient-based saliency analysis, adversarial attacks through synonym substitution, and adversarial retraining as a defense mechanism.

The final classifier achieved approximately 84.7% accuracy on the IMDb test set.

---

## Project Structure

```text
.
├── checkpoints/
├── data/
├── outputs/
├── src/
└── requirements.txt
```

### Main Components

* `train.py` – Train the sentiment classifier.
* `load_model.py` – Load a trained model and perform predictions.
* `visualize_gradients.py` – Generate gradient-based saliency visualizations.
* `attack.py` – Generate adversarial examples using important tokens.
* `defense.py` – Adversarial retraining and robustness evaluation.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

---

## Included Data and Models

The repository already contains:

* Trained model checkpoints (`checkpoints/`)
* IMDb tokenizer (`data/imdb_tokenizer.json`)
* Example saliency visualizations (`outputs/`)

No additional downloads are required.

---

## Running the Classifier

To load the trained model and classify text:

```bash
python3 src/load_model.py
```

Example prediction:

```python
from load_model import load_model, predict

model, tokenizer = load_model()

label, probabilities = predict(
    "This movie was amazing.",
    model,
    tokenizer
)

print(label)
print(probabilities)
```

---

## Training

To train a new classifier from scratch:

```bash
python3 src/train.py
```

The script automatically downloads the IMDb dataset, trains the tokenizer if needed, and saves model checkpoints.

---

## Saliency Visualization

To generate gradient-based token importance visualizations:

```bash
python3 src/visualize_gradients.py
```

Generated visualizations are stored in:

```text
outputs/
```

---

## Adversarial Attacks

To generate adversarial examples:

```bash
python3 src/attack.py
```

The attack identifies influential tokens using saliency information and performs synonym substitutions designed to alter the model prediction while preserving semantic meaning.

---

## Adversarial Defense

To perform adversarial retraining and evaluate robustness:

```bash
python3 src/defense.py
```

---

## Authors

Project Group 10

* Ayda Ghalkhanbaz
* Laasyasree Desu
* Kim Nguyen

DD2417 Language Engineering
KTH Royal Institute of Technology
