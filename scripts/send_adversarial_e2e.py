#!/usr/bin/env python3
"""
End-to-end validation of the live mail stack against the adversarial corpus.

What this does: pulls a stratified sample from one adversarial CSV, sends
each row through the running mail server over authenticated SMTP, waits a
beat for the milter and Sieve rule to settle, then reads the delivered
messages back over IMAPS so we can record what the milter decided and
where Sieve put each message.
Note to reader:
Why a separate script when there are already adversarial CSVs and offline
evaluation results: the offline numbers (attacks/baseline_adversarial_results.csv,
attacks/distilbert_adversarial_results.csv) prove the *models* work. They
say nothing about whether the integrated path SMTP -> milter -> header
injection -> Sieve rule -> mailbox produces those same verdicts on real
delivered mail. scripts/send_milter_tests.sh covers three handcrafted
test messages, but does not cover representative scale. This script is
the bridge between the offline metrics and the live system.

Usage:
    python scripts/send_adversarial_e2e.py \\
        attacks/test_dilution_heavy.csv \\
        evaluation/end_to_end/dilution_heavy_e2e.csv

Optional:
    --n <int>         sample size per class (default 50; total = 2*n)
    --seed <int>      RNG seed (default 42, matches the rest of the project)
    --throttle <sec>  delay between sends (default 0.2)
    --settle <sec>    wait after sending before reading mailbox (default 8)
"""

import argparse
import csv
import datetime
import email
import imaplib
import os
import random
import smtplib
import ssl
import sys
import time
import uuid

# Append-only log of every invocation. Useful for the report write-up
# and as evidence in the OneDrive bundle: one record per run, never
# overwritten, so I can always go back and see what was sent when.
LOG_PATH = "evaluation/end_to_end/run_log.txt"

# Hosts and ports match docker-compose.yml: submission on 587, IMAPS on 993.
SMTP_HOST = "localhost"
SMTP_PORT = 587
IMAP_HOST = "localhost"
IMAP_PORT = 993

# Default credentials match the test users used by send_milter_tests.sh.
# Pulled from env so this script stays in sync if the accounts get
# renamed in postfix-accounts.cf, without having to edit code.
SENDER_USER = os.environ.get("E2E_SENDER_USER", "viserys@mail.local")
SENDER_PASS = os.environ.get("E2E_SENDER_PASS", "viserys123")
RECIPIENT_USER = os.environ.get("E2E_RECIPIENT_USER", "abdulla@mail.local")
RECIPIENT_PASS = os.environ.get("E2E_RECIPIENT_PASS", "abdulla123")


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample(rows, n_per_class, seed):
    """Equal-size split across spam (label=1) and ham (label=0).

    The integration check would be misleading on a 90% spam set, so I
    take n samples from each class regardless of the source CSV's class
    balance. Seeded so the sample is reproducible across runs and on a
    marker's machine.
    """
    rnd = random.Random(seed)
    spam = [r for r in rows if r.get("label") == "1"]
    ham = [r for r in rows if r.get("label") == "0"]
    return (rnd.sample(spam, min(n_per_class, len(spam)))
            + rnd.sample(ham, min(n_per_class, len(ham))))


def relaxed_ssl_context():
    """Permissive SSL context for talking to the local self-signed cert.

    Matches the posture of `swaks --tls` in send_milter_tests.sh. Safe
    here because this script only ever talks to the Docker stack on
    127.0.0.1, where the cert is self-signed and host trust is meaningless.
    Do NOT lift this verbatim into anything that talks to the network.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def send_one(smtp, run_tag, idx, text, label):
    """Submit one message over the already-authenticated SMTP session.

    The email is built as a raw string rather than going through email.mime.
    This is deliberate, and not obvious: MIMEText with an explicit charset
    forces the body to be base64-encoded. The milter's extract_text_from_body
    (milter/spam_milter.py:152) parses raw body bytes only and does not
    see Content-Transfer-Encoding. So a base64'd body arrives at the
    classifier as opaque ASCII gibberish ("VGhpcyBp...") and biases every
    prediction toward spam, which silently breaks every result.

    I lost half a day to this on 2026-04-27 before clocking it; raw-string
    construction is the simplest fix that keeps the body untouched.

    Three custom headers tag the run so the IMAP fetch step can find it:
      X-E2E-Run    per-invocation UUID prefix; isolates concurrent runs
      X-E2E-Index  position in the sampled list, for ordering
      X-E2E-Label  original spam/ham label, for agreement scoring
    """
    payload = (
        f"From: {SENDER_USER}\r\n"
        f"To: {RECIPIENT_USER}\r\n"
        f"Subject: e2e {run_tag} {idx} label={label}\r\n"
        f"X-E2E-Run: {run_tag}\r\n"
        f"X-E2E-Index: {idx}\r\n"
        f"X-E2E-Label: {label}\r\n"
        f"MIME-Version: 1.0\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n"
        f"\r\n"
        f"{text}\r\n"
    )
    smtp.sendmail(SENDER_USER, [RECIPIENT_USER], payload.encode("utf-8"))


def fetch_run(run_tag):
    """Read back every message carrying this run's X-E2E-Run tag.

    Searches INBOX and Junk. The Junk folder matters: the Sieve rule in
    docker/mailserver/config/before.dovecot.sieve routes anything with
    X-Spam-Status: Yes into Junk, so where each message landed is itself
    a piece of evidence. Two things are being verified at once:
      - The milter's headers (X-Spam-Status, X-Spam-Score, X-Spam-Model
        written by milter/spam_milter.py:256-258).
      - That the Sieve rule acted on those headers correctly.

    Mailbox is opened read-only so this script can never accidentally
    mutate mail state, even on a botched re-run.
    """
    delivered = []
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT, ssl_context=relaxed_ssl_context()) as imap:
        imap.login(RECIPIENT_USER, RECIPIENT_PASS)
        for folder in ("INBOX", "Junk"):
            status, _ = imap.select(f'"{folder}"', readonly=True)
            if status != "OK":
                continue
            # IMAP HEADER search (RFC 3501) by our run tag, so two
            # concurrent runs of this script don't contaminate each
            # other's output. Each invocation generates a fresh UUID so
            # the search can only ever match its own messages.
            _, search = imap.search(None, f'(HEADER X-E2E-Run "{run_tag}")')
            ids = search[0].split() if search and search[0] else []
            for num in ids:
                _, raw = imap.fetch(num, "(RFC822)")
                if not raw or not raw[0]:
                    continue
                msg = email.message_from_bytes(raw[0][1])
                delivered.append({
                    "index": msg.get("X-E2E-Index", ""),
                    "label_orig": msg.get("X-E2E-Label", ""),
                    "verdict": msg.get("X-Spam-Status", ""),
                    "score": msg.get("X-Spam-Score", ""),
                    "model": msg.get("X-Spam-Model", ""),
                    "folder": folder,
                })
    return delivered


def summarise(sent, delivered):
    """Two headline integration metrics for the run.

    The dataset-level CSVs already give precision/recall/F1, but they
    can't say anything about routing. These two numbers can:
      - spam_routed_to_junk: did the Sieve rule and milter combine to
        actually put spam-labelled inputs in Junk?
      - ham_routed_to_inbox: did ham-labelled inputs survive both layers
        and reach Inbox?
    Per-message detail lives in the output CSV; this is just the headline.
    """
    n_sent = len(sent)
    n_delivered = len(delivered)
    spam_to_junk = sum(1 for d in delivered if d["label_orig"] == "1" and d["folder"] == "Junk")
    ham_to_inbox = sum(1 for d in delivered if d["label_orig"] == "0" and d["folder"] == "INBOX")
    spam_total = sum(1 for d in delivered if d["label_orig"] == "1")
    ham_total = sum(1 for d in delivered if d["label_orig"] == "0")
    return {
        "sent": n_sent,
        "delivered": n_delivered,
        "spam_routed_to_junk": f"{spam_to_junk}/{spam_total}",
        "ham_routed_to_inbox": f"{ham_to_inbox}/{ham_total}",
    }


def log(msg, log_file):
    """Write the same line to stderr and (if open) the persistent log."""
    print(msg, file=sys.stderr)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()


def main():
    ap = argparse.ArgumentParser(description="End-to-end adversarial validation through the live mail stack.")
    ap.add_argument("input_csv", help="adversarial CSV (text,label,original_text)")
    ap.add_argument("output_csv", help="per-message verdict CSV to write")
    ap.add_argument("--n", type=int, default=50, help="sample size per class (default 50)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--throttle", type=float, default=0.2, help="seconds between sends (default 0.2)")
    ap.add_argument("--settle", type=int, default=8, help="seconds to wait after sending before fetching (default 8)")
    ap.add_argument("--log", default=LOG_PATH, help=f"persistent run log (default {LOG_PATH}, '' to disable)")
    args = ap.parse_args()

    rows = load_csv(args.input_csv)
    sampled = stratified_sample(rows, args.n, args.seed)
    # 8-char UUID is more than enough to disambiguate concurrent runs on
    # a single host. ~16 million possibilities, no coordination needed.
    run_tag = uuid.uuid4().hex[:8]

    # Open the persistent log next to the per-message CSV. Append mode so
    # successive invocations build up a complete history.
    log_file = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log_file = open(args.log, "a", encoding="utf-8")
        log_file.write(f"\n=== {datetime.datetime.now().isoformat(timespec='seconds')} ===\n")

    log(f"[e2e] run_tag={run_tag} input={args.input_csv} sample={len(sampled)}", log_file)

    # One SMTP session for the whole batch. Reusing the auth handshake
    # is faster and closer to how a real client (Outlook, Thunderbird)
    # actually behaves; per-message reconnects would also exercise a
    # different code path on the server side.
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls(context=relaxed_ssl_context())
        smtp.login(SENDER_USER, SENDER_PASS)
        for i, row in enumerate(sampled):
            try:
                send_one(smtp, run_tag, i, row["text"], row["label"])
            except smtplib.SMTPException as e:
                # One bad message shouldn't abort the whole sample. Log
                # the failure and keep going.
                log(f"[e2e] send failed at idx={i}: {e}", log_file)
            time.sleep(args.throttle)

    # Send everything, then settle, then fetch. Doing these interleaved
    # was tempting but doesn't work: the milter dispatch and the Sieve
    # rule are asynchronous, and an immediate fetch returns empty or
    # stale results. The settle window is generous on purpose.
    log(f"[e2e] waiting {args.settle}s for milter + Sieve to settle", log_file)
    time.sleep(args.settle)

    delivered = fetch_run(run_tag)
    log(f"[e2e] fetched {len(delivered)} messages", log_file)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "label_orig", "verdict", "score", "model", "folder"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in delivered:
            w.writerow(row)

    summary = summarise(sampled, delivered)
    log(f"[e2e] summary: {summary}", log_file)
    log(f"[e2e] wrote {args.output_csv}", log_file)

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
