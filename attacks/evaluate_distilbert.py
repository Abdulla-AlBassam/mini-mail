"""
evaluate_distilbert.py: run the DistilBERT model against every adversarial set.

The transformer-side counterpart to evaluate_baseline.py. Loads the saved
fine-tuned DistilBERT, runs predictions over the same nine adversarial CSVs
plus the clean test set, and writes one summary CSV. Then prints a
side-by-side comparison against baseline_adversarial_results.csv so I can
read off the per-attack delta in one place.

Why CPU rather than GPU here: this script also runs inside the spamfilter
container during e2e debugging and that container has no GPU. Letting it
also use CPU on the dev laptop keeps both code paths identical.

Author: Abdulla AlBassam
Module: KV6013, Individual Computing Project
"""

import sys
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Pull in the same preprocessing the model was trained with
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from preprocessing import load_enron_data, preprocess_dataset

# Paths

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'distilbert_spam_classifier')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'enron_spam_data.csv')
ATTACKS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ATTACKS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'distilbert_adversarial_results.csv')
BASELINE_PATH = os.path.join(RESULTS_DIR, 'baseline_adversarial_results.csv')

# 64 fits comfortably in CPU RAM and isn't bottlenecked by tokeniser cost.
# 512 max length so dilution-attack rows (which add words) don't get
# truncated harder than the clean set.
BATCH_SIZE = 64
MAX_LENGTH = 512

# Load model and tokeniser

print('Loading DistilBERT model and tokeniser...')
device = torch.device('cpu')
print(f'  Device: {device}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()
print(f'  Model loaded: {type(model).__name__}')

# Reproduce the exact same split used in training (notebook 03). Numbers
# only line up with the dissertation if random_state and stratify match.

print('\nReproducing clean test set...')
df_raw = load_enron_data(DATA_PATH)
df = preprocess_dataset(df_raw)
_, X_test, _, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
clean_test = pd.DataFrame({'text': X_test.values, 'label': y_test.values})
print(f'  Clean test set: {len(clean_test):,} rows')

# Batched prediction: chunks at BATCH_SIZE so memory stays bounded

def predict_batch(texts):
    """Tokenise and forward each chunk, return (predictions, spam_probabilities)."""
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        spam_probs = probs[:, 1].cpu().numpy()

        all_preds.extend(preds)
        all_probs.extend(spam_probs)

    return np.array(all_preds), np.array(all_probs)

# One row of metrics per (attack, intensity)

def evaluate(texts, labels, attack_name, intensity):
    """Run predictions and compute every metric for one set."""
    text_list = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
    y_pred, y_prob = predict_batch(text_list)
    y_true = np.array(labels)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'attack': attack_name,
        'intensity': intensity,
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall': round(recall_score(y_true, y_pred), 4),
        'f1': round(f1_score(y_true, y_pred), 4),
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }

# Run all evaluations: clean first, then every adversarial CSV in turn

results = []

# 1. Clean test set, the anchor we measure drops from
print('\nEvaluating clean test set...')
row = evaluate(clean_test['text'], clean_test['label'], 'clean', 'none')
results.append(row)
print(f"  Accuracy: {row['accuracy']}  F1: {row['f1']}")

# 2. Every adversarial set in fixed order so the output CSV is stable
adversarial_files = [
    ('char', 'light'),
    ('char', 'medium'),
    ('char', 'heavy'),
    ('synonym', 'light'),
    ('synonym', 'medium'),
    ('synonym', 'heavy'),
    ('dilution', 'light'),
    ('dilution', 'medium'),
    ('dilution', 'heavy'),
]

for attack, intensity in adversarial_files:
    filename = f'test_{attack}_{intensity}.csv'
    filepath = os.path.join(ATTACKS_DIR, filename)
    print(f'Evaluating {filename}...')

    adv_df = pd.read_csv(filepath)
    row = evaluate(adv_df['text'], adv_df['label'], attack, intensity)
    results.append(row)
    print(f"  Accuracy: {row['accuracy']}  F1: {row['f1']}")

# Save the per-set metrics

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f'\nResults saved to: {OUTPUT_PATH}')

# Print a side-by-side comparison against the baseline run

print('\n' + '=' * 95)
print('DISTILBERT vs BASELINE: adversarial evaluation comparison')
print('=' * 95)

# Pull in the baseline numbers to align row-for-row
baseline_df = pd.read_csv(BASELINE_PATH)
baseline_map = {}
for _, r in baseline_df.iterrows():
    key = (r['attack'], r['intensity'])
    baseline_map[key] = r

print(f"{'Attack':<12} {'Intensity':<10} {'BL Acc':>8} {'BL F1':>8} {'DB Acc':>8} {'DB F1':>8} {'Acc Diff':>9} {'F1 Diff':>9}")
print('-' * 95)

for r in results:
    key = (r['attack'], r['intensity'])
    bl = baseline_map.get(key, {})
    bl_acc = bl.get('accuracy', 0) if isinstance(bl, dict) else bl['accuracy']
    bl_f1 = bl.get('f1', 0) if isinstance(bl, dict) else bl['f1']

    acc_diff = r['accuracy'] - bl_acc
    f1_diff = r['f1'] - bl_f1

    print(f"{r['attack']:<12} {r['intensity']:<10} {bl_acc:>8.4f} {bl_f1:>8.4f} {r['accuracy']:>8.4f} {r['f1']:>8.4f} {acc_diff:>+9.4f} {f1_diff:>+9.4f}")

print('-' * 95)

# Per-model drop from its own clean score. This is the headline figure
# in the dissertation: how much ground each model loses to each attack.
db_clean_acc = results[0]['accuracy']
db_clean_f1 = results[0]['f1']
bl_clean_acc = baseline_map[('clean', 'none')]['accuracy']
bl_clean_f1 = baseline_map[('clean', 'none')]['f1']

print(f"\n{'Attack':<12} {'Intensity':<10} {'BL Acc Drop':>12} {'BL F1 Drop':>12} {'DB Acc Drop':>12} {'DB F1 Drop':>12}")
print('-' * 70)
for r in results[1:]:
    key = (r['attack'], r['intensity'])
    bl = baseline_map[key]
    bl_ad = bl_clean_acc - bl['accuracy']
    bl_fd = bl_clean_f1 - bl['f1']
    db_ad = db_clean_acc - r['accuracy']
    db_fd = db_clean_f1 - r['f1']
    print(f"{r['attack']:<12} {r['intensity']:<10} {bl_ad:>+12.4f} {bl_fd:>+12.4f} {db_ad:>+12.4f} {db_fd:>+12.4f}")
print('-' * 70)
print('\nBL = baseline (TF-IDF + LR)    DB = DistilBERT')
