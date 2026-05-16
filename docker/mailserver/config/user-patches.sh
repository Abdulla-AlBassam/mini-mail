#!/bin/bash
# Postfix configuration overrides for the cascade ML spam milter.
# docker-mailserver runs this automatically at container startup
# (https://docker-mailserver.github.io/docker-mailserver/edge/config/advanced/override-defaults/postfix/).
# Source: postconf(5) man page (smtpd_milters, milter_default_action).

# Stack: my spam filter first, then the existing DKIM/DMARC milters that
# docker-mailserver wires up when ENABLE_OPENDKIM=1 / ENABLE_OPENDMARC=1.
# Order matters here: classify FIRST (so we read the body before any
# DKIM signature is added), then DKIM-sign. If we signed first and
# classified after, our X-Spam-* header injection would invalidate the
# signature on every outbound message.
postconf -e 'smtpd_milters = inet:spamfilter:9900, $dkim_milter, $dmarc_milter'
postconf -e 'non_smtpd_milters = $dkim_milter'

# If spamfilter:9900 is unreachable, accept the mail rather than rejecting
# it (fail-open). Better to deliver a possibly-spam message than to bounce
# legitimate mail because the classifier container restarted.
postconf -e "milter_default_action = accept"

# Timeouts. DistilBERT inference is ~50 ms on native x86 but several
# seconds on Apple Silicon under Rosetta, so 180 s on content is generous.
# 30 s on connect/command should be plenty even under load.
postconf -e "milter_connect_timeout = 30s"
postconf -e "milter_command_timeout = 30s"
postconf -e "milter_content_timeout = 180s"

# Protocol 6 is what Postfix and recent libmilter both support.
postconf -e "milter_protocol = 6"
