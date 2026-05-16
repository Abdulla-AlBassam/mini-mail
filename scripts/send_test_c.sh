#!/usr/bin/env bash
# Send only Test C (the borderline phishing case) to deliberately trigger
# DistilBERT escalation. Used to wedge the milter on the OLD code so a
# py-spy stack could be captured for the OMP-deadlock writeup that ended
# up in milter/diagnostics/threading_deadlock.md and Section 3.2.8.
#
# Keeping this script around because: (a) it documents the exact test
# case I used to reproduce the deadlock, and (b) if the deadlock ever
# regresses on a future PyTorch upgrade, this is the fastest way to see
# it happen.
#
# --timeout 300 holds the SMTP connection open for 5 minutes so swaks
# does not give up while the milter is hung. That gives me time to
# `docker exec spamfilter py-spy dump --pid 1` from a second terminal.

set -e

swaks \
  --to abdulla@mail.local \
  --from viserys@mail.local \
  --server localhost \
  --port 587 \
  --auth LOGIN \
  --auth-user viserys@mail.local \
  --auth-password viserys123 \
  -tls \
  --timeout 300 \
  --header 'Subject: Your invoice and account update' \
  --body 'Hello, please find attached your latest invoice. Your account password will need to be updated shortly. Click the secure link to confirm your details. If you did not request this, please ignore this message. Regards, Accounts Team.'
