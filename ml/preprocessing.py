"""
preprocessing.py: shared text cleaning for the Enron spam pipeline.

One module, used by every stage so that training, inference, and adversarial
evaluation all see the same cleaned text. If this drifts, models start
seeing different inputs in training vs production and quality drops without
any obvious failure signal.

Used by: the EDA notebook, the baseline model notebook, the prediction API,
and the adversarial attack scripts.

Author: Abdulla AlBassam
Module: KV6013, Individual Computing Project
"""

import re
import csv
import pandas as pd


def load_enron_data(filepath):
    """
    Read the Enron spam CSV into a DataFrame.

    The Enron dataset has some message bodies that run to ~228 K characters,
    well past Python's default csv field-size limit (which raises an error
    rather than truncating). Bumping the limit to 10^7 covers every row
    I've seen in this dataset and any plausibly-sized email I'd encounter
    later. on_bad_lines='skip' covers a handful of malformed rows that
    pandas otherwise refuses to parse.

    Parameters:
        filepath (str): path to enron_spam_data.csv

    Returns:
        pd.DataFrame: raw dataset with columns
            Message ID, Subject, Message, Spam/Ham, Date
    """
    csv.field_size_limit(10 ** 7)
    df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
    return df


def clean_text(text):
    """
    Clean a single email text string for the classifiers.

    Steps (order matters):
    1. NaN/non-string in goes to "" out (some Enron rows have null bodies).
    2. Lowercase, so "Free", "FREE", "free" all collapse to one token.
    3. Strip leftover email-header lines (From:, To:, etc.). Some rows in
       this dataset still have the raw headers pasted into the message
       body. Without this step the classifier learns to use those headers
       as features, which is meaningless for our use case.
    4. Strip email addresses. They tokenise into noise (random handles,
       random domains) and add nothing the model can generalise from.
    5. Strip URLs. Same reason: random tokens, no useful generalisation,
       and TF-IDF would otherwise build a vocabulary entry for every
       unique link.
    6. Strip HTML tags. A chunk of the Enron spam is HTML-formatted, and
       <br>, <font>, etc. would otherwise dominate the bigram features.
    7. Strip non-alpha characters. Numbers and punctuation tokenise badly
       under TF-IDF; for spam classification on this corpus the alpha
       words carry the signal.
    8. Collapse repeated whitespace into single spaces.

    Two things I deliberately do NOT do here, with reasons:
    - No stemming or lemmatisation. TF-IDF with bigrams already catches
      most morphological variation, and stemming has a habit of merging
      unrelated words ("universal" and "university" both stem to "univers").
    - No stopword removal. That stays at the TF-IDF vectoriser layer where
      it can be toggled per experiment without rewriting this module.

    Parameters:
        text (str): raw email text (subject or body)

    Returns:
        str: cleaned text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = text.lower()

    # Drop email header lines that have been pasted into bodies
    text = re.sub(
        r'^(from|to|cc|bcc|subject|date|sent|received|reply-to|'
        r'content-type|mime-version|x-mailer|x-originating-ip):.*$',
        '', text, flags=re.MULTILINE
    )

    # Drop email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Drop URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Drop HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Drop everything that isn't a letter or space
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Collapse runs of whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def combine_subject_message(row):
    """
    Glue Subject and Message into a single string for the model.

    Both subject and body carry spam signal, and treating them as one
    feature space lets TF-IDF score discriminative terms from either
    location. Subject goes first so its tokens land at the start of
    the document; with sublinear TF this doesn't matter much for the
    baseline, but the DistilBERT pipeline truncates at 256 tokens and
    putting the subject up front guarantees it survives truncation.

    Parameters:
        row (pd.Series): a row from the Enron dataset with 'Subject'
                         and 'Message' columns

    Returns:
        str: combined and cleaned text
    """
    subject = clean_text(row.get('Subject', ''))
    message = clean_text(row.get('Message', ''))
    return f"{subject} {message}".strip()


def preprocess_dataset(df):
    """
    Full preprocessing pipeline for the Enron spam dataset.

    Steps:
    1. Combine Subject + Message into a single 'text' column.
    2. Binary-encode the label: spam = 1, ham = 0.
    3. Drop rows where the combined text is empty after cleaning. This
       loses ~50 rows where the body was empty AND the subject cleaned
       to nothing (often subjects that were just numbers or punctuation).
       Keeping them would feed the model empty strings which it has no
       way to classify.

    Parameters:
        df (pd.DataFrame): raw Enron dataset from load_enron_data()

    Returns:
        pd.DataFrame: cleaned dataset with columns
            - text (str): combined and cleaned email text
            - label (int): 0 for ham, 1 for spam
    """
    df = df.copy()

    # Subject + message in one text column
    df['text'] = df.apply(combine_subject_message, axis=1)

    # Binary label: spam = 1, ham = 0
    df['label'] = (df['Spam/Ham'].str.strip().str.lower() == 'spam').astype(int)

    # Drop rows whose text cleaned to an empty string
    df = df[df['text'].str.len() > 0].reset_index(drop=True)

    return df[['text', 'label']]
