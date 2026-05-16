"""
spam_milter.py:

Two-stage classifier:
1. Every email first goes through the lightweight TF-IDF + Logistic Regression
   baseline.
2. If the baseline's confidence falls below CASCADE_THRESHOLD, the email is
   escalated to the fine-tuned DistilBERT model for a second opinion.

The point of the cascade is to keep DistilBERT off the critical path for the
clear-cut messages (where the baseline is already confident) and only pay the
transformer cost on the borderline cases where it actually changes the answer.

Three headers go on each processed email:
- X-Spam-Status: Yes or No
- X-Spam-Score: spam probability (0.0 to 1.0) from the deciding model
- X-Spam-Model: baseline or distilbert (which model made the final decision)

Dovecot sieve rules sort emails based on X-Spam-Status.

Sources:
- pymilter callback API: https://github.com/sdgathman/pymilter
- Postfix milter protocol: www.postfix.org/MILTER_README.html
- Viola and Jones (2001); "early exit" / "model routing"

Author: Abdulla AlBassam
Module: KV6013
"""

import os
import sys
import logging
import email
from email import policy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import Milter

# Add the ML code directory to the Python path so we can import predict.py
sys.path.insert(0, '/app/ml')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CASCADE_THRESHOLD = float(os.environ.get('CASCADE_THRESHOLD', '0.90'))
MILTER_PORT = int(os.environ.get('MILTER_PORT', '9900'))
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('spam_milter')

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------

BASELINE_MODEL = None
DISTILBERT_MODEL = None
DISTILBERT_AVAILABLE = False

# Every DistilBERT inference goes through this single-worker executor.
# Warmup runs on the same thread, so PyTorch's lazy OMP/MKL init fires once,
# on a thread of our choosing, before any libmilter callback ever asks for
# inference. Without this the first call lands on a libmilter worker thread
# and OMP's first-init path deadlocks under x86_64 emulation (full
# write-up in milter/diagnostics/threading_deadlock.md). The per-call
# timeout then turns any residual hang into a baseline fallback instead of
# letting Postfix block for the full milter_content_timeout (180s).
DISTILBERT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='distilbert')
DISTILBERT_TIMEOUT_SEC = 10.0


def load_models():
    """Load both classifiers into global state. DistilBERT failure is non-fatal."""
    global BASELINE_MODEL, DISTILBERT_MODEL, DISTILBERT_AVAILABLE

    from predict import SpamClassifier, DistilBertSpamClassifier

    log.info("Loading baseline model (TF-IDF + Logistic Regression)...")
    BASELINE_MODEL = SpamClassifier(model_dir='/app/ml/models')
    log.info("Baseline model loaded successfully.")

    try:
        log.info("Loading DistilBERT model...")
        DISTILBERT_MODEL = DistilBertSpamClassifier(
            model_dir='/app/ml/models/distilbert_spam_classifier'
        )
        DISTILBERT_AVAILABLE = True
        log.info("DistilBERT model loaded successfully.")

        log.info("Warming up DistilBERT on dedicated worker thread...")
        DISTILBERT_EXECUTOR.submit(
            DISTILBERT_MODEL.predict,
            "warmup subject",
            "warmup body text for thread pool init",
        ).result(timeout=30)
        log.info("DistilBERT warmup complete.")
    except Exception as e:
        log.warning("DistilBERT failed to load: %s. Using baseline only.", e)
        DISTILBERT_AVAILABLE = False

    log.info(
        "Cascade threshold: %.2f | DistilBERT available: %s",
        CASCADE_THRESHOLD, DISTILBERT_AVAILABLE
    )


# ---------------------------------------------------------------------------
# Cascade classification
# ---------------------------------------------------------------------------

def classify_email(subject, body_text):
    """
    Run the cascade classifier on a single email.

    Returns:
        tuple: (label, spam_proba, model_used)
            - label: 'spam' or 'ham'
            - spam_proba: probability of spam from the deciding model (0.0 to 1.0)
            - model_used: 'baseline' or 'distilbert'
    """
    # Stage 1: baseline
    label, spam_proba = BASELINE_MODEL.predict(subject, body_text)
    confidence = max(spam_proba, 1.0 - spam_proba)

    if confidence >= CASCADE_THRESHOLD or not DISTILBERT_AVAILABLE:
        return label, spam_proba, 'baseline'

    # Stage 2: hand off to DistilBERT on the dedicated worker thread, so
    # every real inference runs on the same thread that ran the in warmup.
    log.info(
        "Escalating to DistilBERT (baseline confidence=%.4f, threshold=%.2f)",
        confidence, CASCADE_THRESHOLD
    )
    try:
        future = DISTILBERT_EXECUTOR.submit(
            DISTILBERT_MODEL.predict, subject, body_text
        )
        label, spam_proba = future.result(timeout=DISTILBERT_TIMEOUT_SEC)
    except FutureTimeout:
        log.warning(
            "DistilBERT timeout after %.1fs; falling back to baseline verdict",
            DISTILBERT_TIMEOUT_SEC,
        )
        return label, spam_proba, 'baseline_timeout_fallback'
    return label, spam_proba, 'distilbert'


# ---------------------------------------------------------------------------
# Email body extraction
# ---------------------------------------------------------------------------

def extract_text_from_body(raw_body):
    """
    Pull plain text out of the raw email body bytes.

    For multipart messages, walk parts and grab the first text/plain.
    If there isn't one, fall back to text/html (better than nothing for
    classification). If MIME parsing itself blows up, treat the whole
    body as raw text. Empty string on no usable content.
    """
    try:
        msg = email.message_from_bytes(raw_body, policy=policy.default)

        # Multipart: pick the text/plain part first
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    payload = part.get_content()
                    if isinstance(payload, str):
                        return payload
            # No text/plain anywhere, try HTML
            for part in msg.walk():
                if part.get_content_type() == 'text/html':
                    payload = part.get_content()
                    if isinstance(payload, str):
                        return payload
            return ''
        else:
            payload = msg.get_content()
            if isinstance(payload, str):
                return payload
            return ''
    except Exception:
        # MIME parser failed, treat the whole body as raw text
        return raw_body.decode('utf-8', errors='replace')


# ---------------------------------------------------------------------------
# Milter class
# ---------------------------------------------------------------------------

class CascadeSpamMilter(Milter.Base):
    """
    Postfix milter that runs the cascade classifier on incoming email.

    Per-message state is reset in envfrom() and accumulated by the header()
    and body() callbacks. Classification only happens in eom() once the
    whole message is in hand.
    """

    def __init__(self):
        self.id = Milter.uniqueID()
        self.subject = ''
        self.body_chunks = []
        self.headers = {}

    @Milter.noreply
    def connect(self, hostname, family, hostaddr):
        return Milter.CONTINUE

    @Milter.noreply
    def hello(self, heloname):
        return Milter.CONTINUE

    def envfrom(self, mailfrom, *args):
        """Reset per-message state for each new email."""
        self.subject = ''
        self.body_chunks = []
        self.headers = {}
        return Milter.CONTINUE

    @Milter.noreply
    def envrcpt(self, to, *args):
        return Milter.CONTINUE

    @Milter.noreply
    def header(self, name, value):
        """Capture the Subject header for classification."""
        self.headers[name.lower()] = value
        if name.lower() == 'subject':
            self.subject = value
        return Milter.CONTINUE

    @Milter.noreply
    def body(self, chunk):
        """Accumulate body chunks (may be called multiple times per email)."""
        self.body_chunks.append(chunk)
        return Milter.CONTINUE

    def eom(self):
        """
        End of message: classify and stamp headers.

        Always returns ACCEPT. Routing into Junk is the sieve rule's job,
        not the milter's. If classification blows up for any reason, the
        email is delivered without spam headers (fail-open). Better that
        than holding mail hostage on a classifier bug.
        """
        try:
            raw_body = b''.join(self.body_chunks)
            body_text = extract_text_from_body(raw_body)

            label, spam_proba, model_used = classify_email(
                self.subject, body_text
            )

            status = 'Yes' if label == 'spam' else 'No'

            self.addheader('X-Spam-Status', status)
            self.addheader('X-Spam-Score', f'{spam_proba:.4f}')
            self.addheader('X-Spam-Model', model_used)

            log.info(
                "id=%s subject=\"%.50s\" status=%s score=%.4f model=%s",
                self.id, self.subject, status, spam_proba, model_used
            )

        except Exception as e:
            log.error("Classification failed for id=%s: %s", self.id, e)
            # Fail-open: deliver without spam headers rather than block mail

        return Milter.ACCEPT

    def close(self):
        """Cleanup per-connection state."""
        self.body_chunks = []
        return Milter.CONTINUE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Starting cascade spam milter on port %d...", MILTER_PORT)

    load_models()

    # Hand the milter class to libmilter and start the event loop.
    # ADDHDRS is the only flag we need: we add headers, never modify or
    # remove anything from the message.
    Milter.factory = CascadeSpamMilter
    Milter.set_flags(Milter.ADDHDRS)

    listen_addr = f"inet:{MILTER_PORT}"
    log.info("Milter listening on %s", listen_addr)
    Milter.runmilter('CascadeSpamMilter', listen_addr, timeout=300)

    log.info("Milter shutdown.")


if __name__ == '__main__':
    main()
