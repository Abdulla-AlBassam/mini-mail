"""
predict.py: inference API for the two spam classifiers.

Two classifier classes that share the same interface so the milter and the
attack scripts can swap between them without caring which is which:
- SpamClassifier: TF-IDF + Logistic Regression baseline
- DistilBertSpamClassifier: fine-tuned DistilBERT transformer

Both expose predict(subject, body) and predict_batch(texts).
Use get_classifier(model_type) when the choice of model is decided at runtime
rather than at import time.

Used by:
- Phase 5: mail server integration (classify incoming emails in the milter)
- Phase 6: adversarial attack evaluation (classify perturbed emails in batch)

Author: Abdulla AlBassam
Module: KV6013, Individual Computing Project
"""

import os
import joblib
from preprocessing import clean_text


class SpamClassifier:
    """
    Wrapper around the trained baseline spam classifier.

    Loads the saved TF-IDF vectoriser and Logistic Regression model and
    exposes a single predict(subject, body) method so callers don't have
    to care about preprocessing or feature extraction. The two .pkl
    files come from notebook 02_baseline_model.ipynb.

    Usage:
        clf = SpamClassifier()
        label, confidence = clf.predict("Free money!", "Click here now")
        # ('spam', 0.95)
    """

    def __init__(self, model_dir=None):
        """
        Load the trained model and vectoriser.

        Parameters:
            model_dir (str, optional): Path to directory containing
                baseline_logistic_regression_model.pkl and baseline_tfidf_vectorizer.pkl.
                Defaults to ml/models/ relative to this file.
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')

        model_path = os.path.join(model_dir, 'baseline_logistic_regression_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'baseline_tfidf_vectorizer.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 02_baseline_model.ipynb first to train and save the model."
            )
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectoriser not found at {vectorizer_path}. "
                "Run 02_baseline_model.ipynb first to train and save the vectoriser."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, subject, body):
        """
        Classify a single email.

        Runs the same preprocessing the training pipeline used: clean_text
        on subject and body separately, then concatenate. Critical that
        this matches preprocessing.preprocess_dataset(); otherwise inference
        text drifts from training text and the model degrades silently.

        Parameters:
            subject (str): Email subject line
            body (str): Email message body

        Returns:
            tuple: (label, confidence)
                - label (str): 'spam' or 'ham'
                - confidence (float): probability of spam (0.0 to 1.0)
        """
        text = f"{clean_text(subject)} {clean_text(body)}".strip()
        features = self.vectorizer.transform([text])
        proba = self.model.predict_proba(features)[0, 1]
        label = 'spam' if proba >= 0.5 else 'ham'
        return label, float(proba)

    def predict_batch(self, texts):
        """
        Classify many already-preprocessed text strings at once.

        Used by the adversarial attack scripts where the texts have already
        been cleaned and we just want fast batched inference. Don't call
        this with raw email subject + body, the texts won't have been run
        through clean_text and you'll get wrong predictions.

        Parameters:
            texts (list[str]): list of preprocessed text strings

        Returns:
            list[tuple]: list of (label, confidence) tuples
        """
        features = self.vectorizer.transform(texts)
        probas = self.model.predict_proba(features)[:, 1]
        labels = ['spam' if p >= 0.5 else 'ham' for p in probas]
        return list(zip(labels, probas.tolist()))


class DistilBertSpamClassifier:
    """
    Wrapper around the fine-tuned DistilBERT spam classifier.

    Loads the saved model and tokeniser from distilbert_spam_classifier/
    and exposes the same interface as SpamClassifier so callers can use
    either interchangeably.

    Note: transformers and torch are imported inside the methods (lazy
    imports) so anyone who only needs the baseline classifier doesn't
    have to install ~1 GB of PyTorch wheels just to satisfy an import.
    Same reason transformers is in requirements.txt as optional in spirit
    even though pip will install it.

    Usage:
        clf = DistilBertSpamClassifier()
        label, confidence = clf.predict("Free money!", "Click here now")
        # ('spam', 0.98)
    """

    def __init__(self, model_dir=None):
        """
        Load the fine-tuned DistilBERT model and tokeniser.

        Parameters:
            model_dir (str, optional): Path to directory containing the
                saved model (config.json, model.safetensors, tokeniser files).
                Defaults to ml/models/distilbert_spam_classifier/ relative to this file.
        """
        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(__file__), 'models', 'distilbert_spam_classifier'
            )

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"DistilBERT model not found at {model_dir}. "
                "Run 03_distilbert_model.ipynb on Google Colab first, "
                "then download and place the model files in ml/models/distilbert_spam_classifier/"
            )

        # Lazy imports: only paid for when the DistilBERT classifier is
        # actually instantiated, so baseline-only callers don't pull torch.
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        import torch

        self.torch = torch
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.max_length = 256

    def predict(self, subject, body):
        """
        Classify a single email.

        Same preprocessing pipeline as the baseline (clean_text on subject
        and body, then concatenate) before WordPiece tokenisation. Keeping
        these aligned matters for the cascade: if the two models see
        different text, the threshold between them stops being meaningful.

        Parameters:
            subject (str): Email subject line
            body (str): Email message body

        Returns:
            tuple: (label, confidence)
                - label (str): 'spam' or 'ham'
                - confidence (float): probability of spam (0.0 to 1.0)
        """
        from scipy.special import softmax

        text = f"{clean_text(subject)} {clean_text(body)}".strip()
        inputs = self.tokenizer(
            text, truncation=True, padding=True,
            max_length=self.max_length, return_tensors='pt'
        ).to(self.device)

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]

        proba = softmax(logits)[1]  # spam = class 1
        label = 'spam' if proba >= 0.5 else 'ham'
        return label, float(proba)

    def predict_batch(self, texts):
        """
        Classify many already-preprocessed text strings at once.

        Chunks at 32 to keep peak GPU memory predictable; the T4 used for
        training has 15 GB but inference on CPU in the milter container is
        much tighter. 32 is empirical, not principled, but it never OOM'd
        during the adversarial evaluation runs.

        Parameters:
            texts (list[str]): list of preprocessed text strings

        Returns:
            list[tuple]: list of (label, confidence) tuples
        """
        from scipy.special import softmax
        import numpy as np

        all_labels = []
        all_probas = []
        chunk_size = 32

        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            inputs = self.tokenizer(
                chunk, truncation=True, padding=True,
                max_length=self.max_length, return_tensors='pt'
            ).to(self.device)

            with self.torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu().numpy()

            probas = softmax(logits, axis=1)[:, 1]
            labels = ['spam' if p >= 0.5 else 'ham' for p in probas]
            all_labels.extend(labels)
            all_probas.extend(probas.tolist())

        return list(zip(all_labels, all_probas))


def get_classifier(model_type='baseline'):
    """
    Factory: pick a classifier by string name.

    Lets calling code accept the model choice from a config file or env
    var without having to import both classes (which would force the
    transformers/torch import path even when the user only wants the
    baseline).

    Parameters:
        model_type (str): 'baseline' for TF-IDF + LR, 'distilbert' for DistilBERT

    Returns:
        SpamClassifier or DistilBertSpamClassifier instance

    Raises:
        ValueError: if model_type is not 'baseline' or 'distilbert'
    """
    if model_type == 'baseline':
        return SpamClassifier()
    elif model_type == 'distilbert':
        return DistilBertSpamClassifier()
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Use 'baseline' or 'distilbert'."
        )
