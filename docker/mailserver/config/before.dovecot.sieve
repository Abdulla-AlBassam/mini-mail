require ["fileinto", "mailbox"];

# Route anything stamped X-Spam-Status: Yes by the cascade milter into the
# Junk folder. The header is set by milter/spam_milter.py:256, and the
# milter is the only thing on this stack that writes it, so a header
# match here is equivalent to the milter's verdict.
# Source: Dovecot Pigeonhole Sieve documentation (fileinto, mailbox extensions)

if header :contains "X-Spam-Status" "Yes" {
    fileinto :create "Junk";
    stop;
}
