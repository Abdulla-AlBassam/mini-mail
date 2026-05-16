"""
evaluate_baseline.py: run the baseline classifier against every adversarial set.

Loads the saved TF-IDF vectoriser and Logistic Regression model, then scores
the original clean test set plus the nine adversarial CSVs (3 attacks x 3
intensities). Writes one summary CSV with all the metrics so the report
draws from a single source of truth.

Mirrored by evaluate_distilbert.py, which runs the same evaluation against
the transformer model. Keeping them parallel matters: the comparison in
the dissertation is only meaningful if both models see the same texts and
metrics in the same order.

Author: Abdulla AlBassam
Module: KV6013, Individual Computing Project
"""

import sys
import os
import joblib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

# Reach into the sibling ml/ directory so we share the exact same
# preprocessing as training. Hard requirement: any drift here means we
# evaluate on a different distribution than the model was trained on.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from preprocessing import load_enron_data, preprocess_dataset

# Paths

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'baseline_logistic_regression_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'models', 'baseline_tfidf_vectorizer.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'enron_spam_data.csv')
ATTACKS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ATTACKS_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'baseline_adversarial_results.csv')

# Load the saved baseline model

print('Loading model and vectoriser...')
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print('  Model loaded:', type(model).__name__)
print('  Vectoriser loaded:', type(vectorizer).__name__)

# Reproduce the exact 80/20 split the model was trained on. random_state
# and stratify must match notebook 02 byte-for-byte; otherwise the "clean"
# numbers here won't match the dissertation's reported baseline.

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

# Score one set of texts and return a metrics dict ready for the summary CSV

def evaluate(texts, labels, attack_name, intensity):
    """Vectorise, predict, compute all metrics, return one row."""
    X = vectorizer.transform(texts)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    return {
        'attack': attack_name,
        'intensity': intensity,
        'accuracy': round(accuracy_score(labels, y_pred), 4),
        'precision': round(precision_score(labels, y_pred), 4),
        'recall': round(recall_score(labels, y_pred), 4),
        'f1': round(f1_score(labels, y_pred), 4),
        'roc_auc': round(roc_auc_score(labels, y_prob), 4),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }

# Run evaluations: clean set first, then every adversarial CSV in turn

results = []

# 1. Clean test set, our baseline anchor
print('\nEvaluating clean test set...')
row = evaluate(clean_test['text'], clean_test['label'], 'clean', 'none')
results.append(row)
print(f"  Accuracy: {row['accuracy']}  F1: {row['f1']}")

# 2. Every adversarial set, in a fixed order so the output CSV is stable
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

# Save the per-set metrics to one CSV

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f'\nResults saved to: {OUTPUT_PATH}')

# Print a console-friendly summary table for quick eyeballing

print('\n' + '=' * 75)
print('BASELINE MODEL: adversarial evaluation summary')
print('=' * 75)
print(f"{'Attack':<12} {'Intensity':<10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print('-' * 75)
for r in results:
    print(f"{r['attack']:<12} {r['intensity']:<10} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
print('-' * 75)

# Drop from clean baseline. This is the headline number: how much each
# attack degrades accuracy/F1 relative to the model's own clean score.
clean_acc = results[0]['accuracy']
clean_f1 = results[0]['f1']
print(f"\n{'Attack':<12} {'Intensity':<10} {'Acc Drop':>10} {'F1 Drop':>10}")
print('-' * 45)
for r in results[1:]:
    acc_drop = clean_acc - r['accuracy']
    f1_drop = clean_f1 - r['f1']
    print(f"{r['attack']:<12} {r['intensity']:<10} {acc_drop:>+10.4f} {f1_drop:>+10.4f}")
print('-' * 45)
