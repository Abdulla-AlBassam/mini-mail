"""
generate_adversarial.py: build the nine adversarial test sets.

Applies three attack types at three intensities to the spam half of the
test set. Ham is left alone, because attacking ham would just make the
ham-as-ham F1 worse and isn't what an attacker would actually do.

Attacks:
    1. Character-level perturbation: swap / insert / delete / substitute
       a fraction of internal characters in spam words. Models a typo-style
       evasion and the easiest one to automate.
    2. Synonym substitution: replace content words with WordNet synonyms.
       Models an attacker rephrasing the trigger words.
    3. Text dilution: inject benign professional words at random positions.
       Models the classic "good-word injection" attack against bag-of-words
       classifiers.

Output: nine CSVs (test_<attack>_<intensity>.csv) plus one attack_summary.csv
that records perturbation rates per set. Reproducible: every attack is
seeded from a fixed function of (attack_name, intensity) so the same
(attack, intensity) pair always produces the same set, while different
pairs don't share state.

Author: Abdulla AlBassam
Module: KV6013, Individual Computing Project
"""

import sys
import os
import random
import string
import csv

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import train_test_split

# Pull in the same preprocessing pipeline the model was trained on
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml'))
from preprocessing import load_enron_data, preprocess_dataset

# Configuration

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'ml', 'data', 'raw', 'enron_spam_data.csv')
OUTPUT_DIR = os.path.dirname(__file__)

STOPWORDS = set(stopwords.words('english'))

# Intensities chosen so 'light' is plausibly invisible to a human reader,
# 'medium' is detectable on a careful read, and 'heavy' is obviously
# tampered with. Char rates are fractions of internal characters; synonym
# rates are fractions of total words; dilution is an absolute injection
# count (not a rate, since it's adding rather than perturbing).
CHAR_RATES = {'light': 0.05, 'medium': 0.10, 'heavy': 0.20}
SYNONYM_RATES = {'light': 0.01, 'medium': 0.03, 'heavy': 0.05}
DILUTION_COUNTS = {'light': 3, 'medium': 7, 'heavy': 12}

# Vocabulary used for the dilution attack. Every word here is something
# you'd see in a normal corporate email; the classic good-word-injection
# trick is to dilute spammy text with high-frequency ham vocab so that the
# average TF-IDF score moves toward "ham".
BENIGN_WORDS = [
    'meeting', 'regards', 'schedule', 'report', 'please',
    'attached', 'update', 'confirm', 'agenda', 'appreciated',
    'review', 'project', 'team', 'discuss', 'proposal',
    'deadline', 'summary', 'feedback', 'approved', 'quarterly'
]

# Penn Treebank tags we treat as "content words" for the synonym attack.
# Skipping NNP/NNPS deliberately so we don't replace proper nouns with
# WordNet synonyms (which would mangle names and produce nonsense).
CONTENT_POS = {'NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
               'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}


# Attack 1: character-level perturbation

def char_swap(word, idx):
    """Swap character at idx with the next character."""
    if idx >= len(word) - 1:
        return word
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return ''.join(chars)


def char_insert(word, idx):
    """Insert a random letter at position idx."""
    chars = list(word)
    chars.insert(idx, random.choice(string.ascii_lowercase))
    return ''.join(chars)


def char_delete(word, idx):
    """Delete character at position idx."""
    chars = list(word)
    del chars[idx]
    return ''.join(chars)


def char_substitute(word, idx):
    """Replace character at idx with a random different letter."""
    chars = list(word)
    replacement = random.choice([c for c in string.ascii_lowercase if c != chars[idx]])
    chars[idx] = replacement
    return ''.join(chars)


def perturb_word(word):
    """Pick one internal character of word and apply a random op to it.

    Only touches internal characters (not first or last) because most
    readers can absorb internal scrambling and still read the word, but
    a tokeniser/vectoriser sees a different token. Words of 2 or fewer
    characters are skipped: there's no internal position to edit.
    """
    if len(word) <= 2:
        return word
    idx = random.randint(1, len(word) - 2)
    op = random.choice([char_swap, char_insert, char_delete, char_substitute])
    return op(word, idx)


def attack_char_perturbation(text, rate):
    """Perturb roughly `rate` fraction of internal characters in `text`.

    Strategy: count perturbable characters across all words, decide how
    many edits we need, then pick distinct word positions to edit (one
    edit per word maximum). One-edit-per-word matters because two edits
    in the same word can collide with each other and inflate the
    character-diff rate beyond `rate`.
    """
    words = text.split()
    if not words:
        return text

    # Internal-character count per word (max(0, len-2) skips short words)
    internal_char_counts = []
    for w in words:
        count = max(0, len(w) - 2)
        internal_char_counts.append(count)

    total_internal = sum(internal_char_counts)
    if total_internal == 0:
        return text

    num_edits = max(1, int(total_internal * rate))

    # Build a flat list mapping each internal-character slot to its word.
    # Sampling across this list weights selection by word length, so long
    # words get edited more often than short ones (which feels right:
    # we want edits distributed by character mass, not by word count).
    char_to_word = []
    for w_idx, w in enumerate(words):
        for c_idx in range(1, max(1, len(w) - 1)):
            if len(w) > 2:
                char_to_word.append(w_idx)

    if not char_to_word:
        return text

    # Walk the shuffled list and accumulate distinct word indices until
    # we have num_edits of them.
    selected_words = set()
    indices = list(range(len(char_to_word)))
    random.shuffle(indices)
    for idx in indices:
        if len(selected_words) >= num_edits:
            break
        w_idx = char_to_word[idx]
        if w_idx not in selected_words:
            selected_words.add(w_idx)

    # Apply one perturbation to each selected word
    result = []
    for i, w in enumerate(words):
        if i in selected_words:
            result.append(perturb_word(w))
        else:
            result.append(w)

    return ' '.join(result)


# Attack 2: synonym substitution

def get_wordnet_pos(treebank_tag):
    """Map a Penn Treebank tag to WordNet's coarser POS set."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return None


def get_synonym(word, pos_tag_str):
    """Find a single-word WordNet synonym for `word`, or None.

    Single-word only (no underscores, no spaces). Multi-word lemmas would
    inflate the word count and break the assumption that synonym
    substitution preserves length, which would in turn mess up
    word_diff_rate's positional comparison downstream.
    """
    wn_pos = get_wordnet_pos(pos_tag_str)
    if wn_pos is None:
        return None

    synsets = wordnet.synsets(word, pos=wn_pos)
    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name()
            if '_' in name or ' ' in name:
                continue
            if name.lower() != word.lower():
                synonyms.add(name)

    if not synonyms:
        return None
    return random.choice(list(synonyms))


def attack_synonym_substitution(text, rate):
    """Replace `rate` fraction of content words with WordNet synonyms.

    Eligibility filters:
    - Must be a content POS (noun/verb/adj/adv). We don't want to swap
      stopwords like "the" or "is" with random synonyms.
    - Not in the NLTK stopword list.
    - At least 4 characters: short words rarely have meaningful synonyms
      and swapping them tends to produce noise.
    - Not proper nouns (NNP/NNPS): see CONTENT_POS comment above.
    """
    words = text.split()
    if not words:
        return text

    # POS-tag once up front rather than per-iteration
    tagged = pos_tag(words)

    # Pick out the indices of words that pass every eligibility filter
    eligible_indices = []
    for i, (word, tag) in enumerate(tagged):
        if (tag in CONTENT_POS
                and word.lower() not in STOPWORDS
                and len(word) >= 4
                and tag not in ('NNP', 'NNPS')):
            eligible_indices.append(i)

    total_words = len(words)
    num_replacements = max(1, int(total_words * rate)) if eligible_indices else 0
    num_replacements = min(num_replacements, len(eligible_indices))

    selected = random.sample(eligible_indices, num_replacements) if num_replacements > 0 else []
    selected_set = set(selected)

    result = list(words)
    for i in selected_set:
        word, tag = tagged[i]
        synonym = get_synonym(word, tag)
        if synonym is not None:
            result[i] = synonym

    return ' '.join(result)


# Attack 3: text dilution (benign word injection)

def attack_text_dilution(text, num_words):
    """Inject `num_words` benign words at random positions inside `text`.

    Each insertion target is sampled fresh from len(words) (post-insertion
    length grows each iteration), so injections naturally spread across
    the message rather than clustering at one point.
    """
    words = text.split()
    if not words:
        return text

    for _ in range(num_words):
        inject_word = random.choice(BENIGN_WORDS)
        pos = random.randint(0, len(words))
        words.insert(pos, inject_word)

    return ' '.join(words)


# Perturbation rate measurement
#
# These functions exist so the attack_summary CSV reports the *actual*
# perturbation rate per row, not just the requested rate. The reported
# rate is what readers should trust; the requested rate is just a config
# knob.

def char_diff_rate(original, perturbed):
    """Fraction of characters changed, measured word-pair by word-pair.

    The character attack preserves word count, so we can pair words by
    position and count edits inside each changed pair. A naive string
    comparison would over-count: a single insertion shifts every later
    character, making the diff look much larger than the actual edit
    distance.
    """
    orig_words = original.split()
    pert_words = perturbed.split()
    total_chars = sum(len(w) for w in orig_words)
    if total_chars == 0:
        return 0.0

    chars_changed = 0
    for ow, pw in zip(orig_words, pert_words):
        if ow != pw:
            # Count character-level differences within this word pair
            for a, b in zip(ow, pw):
                if a != b:
                    chars_changed += 1
            chars_changed += abs(len(ow) - len(pw))

    return chars_changed / total_chars


def word_diff_rate(original, perturbed):
    """Fraction of words replaced (changed words / original word count).

    Single-word synonyms only (see get_synonym), so word count is preserved
    and a simple positional zip works. If we ever allow multi-word
    synonyms, this comparison breaks and we'd need to switch to an
    alignment-based metric.
    """
    orig_words = original.split()
    pert_words = perturbed.split()
    if len(orig_words) == 0:
        return 0.0
    changed = sum(1 for a, b in zip(orig_words, pert_words) if a != b)
    return changed / len(orig_words)


def injection_count(original, perturbed):
    """How many words were added (used by the dilution attack only)."""
    return len(perturbed.split()) - len(original.split())


# Main pipeline

def load_test_set():
    """Reproduce the same 80/20 stratified test split the notebooks use."""
    print('Loading and preprocessing dataset...')
    df_raw = load_enron_data(DATA_PATH)
    df = preprocess_dataset(df_raw)
    print(f'  Dataset: {len(df):,} rows')

    _, X_test, _, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    test_df = pd.DataFrame({'text': X_test.values, 'label': y_test.values})
    print(f'  Test set: {len(test_df):,} rows '
          f'(Ham: {(test_df["label"] == 0).sum()}, Spam: {(test_df["label"] == 1).sum()})')
    return test_df


def apply_attack(test_df, attack_fn, attack_name, intensity_name, attack_param):
    """Run one attack at one intensity over every spam row in the test set.

    Re-seeds RNG from a hash of (attack_name, intensity_name) so two
    different runs of this script produce identical CSVs (reproducibility),
    while different (attack, intensity) pairs are independent. Without the
    re-seed, the dilution_heavy run would inherit RNG state from the
    char_heavy run before it, and changing the order of attacks in the
    list below would change every output file.
    """
    print(f'  Applying {attack_name} ({intensity_name})...')

    result = test_df.copy()
    result['original_text'] = result['text']

    spam_mask = result['label'] == 1
    spam_texts = result.loc[spam_mask, 'text'].values

    random.seed(SEED + hash(f'{attack_name}_{intensity_name}') % (2**31))
    np.random.seed(SEED + hash(f'{attack_name}_{intensity_name}') % (2**31))

    perturbed = []
    for text in spam_texts:
        perturbed.append(attack_fn(text, attack_param))

    result.loc[spam_mask, 'text'] = perturbed
    return result


def compute_summary_stats(test_df, result_df, attack_name, intensity_name, attack_type):
    """Build one row for attack_summary.csv describing this attack run."""
    spam_mask = result_df['label'] == 1
    originals = result_df.loc[spam_mask, 'original_text'].values
    perturbed = result_df.loc[spam_mask, 'text'].values

    num_modified = sum(1 for o, p in zip(originals, perturbed) if o != p)

    if attack_type == 'char':
        rates = [char_diff_rate(o, p) for o, p in zip(originals, perturbed)]
        avg_rate = np.mean(rates)
        metric_name = 'avg_char_diff_rate'
    elif attack_type == 'synonym':
        rates = [word_diff_rate(o, p) for o, p in zip(originals, perturbed)]
        avg_rate = np.mean(rates)
        metric_name = 'avg_word_diff_rate'
    else:  # dilution
        counts = [injection_count(o, p) for o, p in zip(originals, perturbed)]
        avg_rate = np.mean(counts)
        metric_name = 'avg_words_injected'

    return {
        'attack': attack_name,
        'intensity': intensity_name,
        'total_spam': int(spam_mask.sum()),
        'spam_modified': num_modified,
        'modification_pct': round(num_modified / spam_mask.sum() * 100, 2),
        'metric': metric_name,
        'metric_value': round(avg_rate, 4)
    }


def main():
    test_df = load_test_set()
    summary_rows = []

    attacks = [
        # (name, short_name, attack_fn, param_dict, attack_type)
        ('Character Perturbation', 'char', attack_char_perturbation, CHAR_RATES, 'char'),
        ('Synonym Substitution', 'synonym', attack_synonym_substitution, SYNONYM_RATES, 'synonym'),
        ('Text Dilution', 'dilution', attack_text_dilution, DILUTION_COUNTS, 'dilution'),
    ]

    for attack_name, short_name, attack_fn, param_dict, attack_type in attacks:
        print(f'\n--- {attack_name} ---')
        for intensity, param in param_dict.items():
            result_df = apply_attack(test_df, attack_fn, short_name, intensity, param)

            # Save adversarial CSV
            filename = f'test_{short_name}_{intensity}.csv'
            filepath = os.path.join(OUTPUT_DIR, filename)
            result_df[['text', 'label', 'original_text']].to_csv(filepath, index=False)
            print(f'    Saved: {filename} ({len(result_df):,} rows)')

            # Compute stats
            stats = compute_summary_stats(test_df, result_df, attack_name, intensity, attack_type)
            stats['filename'] = filename
            summary_rows.append(stats)

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, 'attack_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary saved: attack_summary.csv')
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
