#!/usr/bin/env bash
# Send three handcrafted test emails through the local mailserver
# on the authenticated submission port (587). Goal is to exercise the
# cascade ML milter end-to-end with one obvious ham, one obvious spam,
# and one borderline case that I expect the baseline to be unsure about
# and the cascade to escalate to DistilBERT.
#
# Auth is required because the laptop's IP is not the mailserver IP,
# so SPF would otherwise reject. An authenticated submission bypasses
# that check, which is what real users would also do via Roundcube.

set -e

SERVER=localhost
PORT=587
FROM=viserys@mail.local
TO=abdulla@mail.local
USER=viserys@mail.local
PASS=viserys123

COMMON=(
  --to "$TO"
  --from "$FROM"
  --server "$SERVER"
  --port "$PORT"
  --auth LOGIN
  --auth-user "$USER"
  --auth-password "$PASS"
  -tls
)

echo
echo "=== Test A: obvious ham ==="
swaks "${COMMON[@]}" \
  --header 'Subject: Meeting tomorrow' \
  --body 'Hi Abdulla, can we move our 1:1 to 3pm tomorrow? I have a clash with the supervisor meeting. Cheers, Viserys'

echo
echo "=== Test B: obvious spam ==="
swaks "${COMMON[@]}" \
  --header 'Subject: CONGRATULATIONS YOU WON 1000000 DOLLARS' \
  --body 'Dear winner, you have been selected to receive ONE MILLION DOLLARS in our international lottery. Click http://claim-prize-now.example to claim your prize money. Send your bank details, full name and passport copy to verify. Free Viagra included. Act now, limited time offer.'

echo
echo "=== Test C: borderline ==="
swaks "${COMMON[@]}" \
  --header 'Subject: Your invoice and account update' \
  --body 'Hello, please find attached your latest invoice. Your account password will need to be updated shortly. Click the secure link to confirm your details. If you did not request this, please ignore this message. Regards, Accounts Team.'

echo
echo "All three submitted. Now run: docker compose logs --tail 80 spamfilter"
