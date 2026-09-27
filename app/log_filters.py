"""What happens to a log record before it is written.

Two filters, installed on the ROOT handlers so they cover every logger in the process —
including uvicorn's, whose access lines are the ones carrying a URL.

`RequestIdFilter` lives in `correlation`, with the concept it belongs to. What is here is the
redaction: a rule about log output, not about the thing being redacted.
"""

from __future__ import annotations

import logging
import re

# `/share/<id>` with whatever follows it on the line. The id is a bearer capability — the
# whole authorization, valid for up to a week — so it is the same class of secret as the API
# key, which this app already keeps out of logs, URLs and error bodies.
_SHARE_PATH = re.compile(r"(/share/)[^\s\"?]+(\?[^\s\"]*)?")


class RedactShareTokens(logging.Filter):
    """Replace a share id in a log line with ``<redacted>`` (audit SEC-9).

    uvicorn runs with access logging on, so every download wrote its capability to the log:
    anyone with log access held every outstanding link until it expired. Turning the access
    log off would have cost the record of *which* routes were hit, so the line is kept and
    the secret in it is not — the same trade as sanitizing a 500 body while logging the
    traceback.

    Rewrites ``record.args`` rather than ``record.msg``: uvicorn's access record is a format
    string plus a tuple, and the path is one of the arguments, so touching the message alone
    would leave the id to be substituted in at write time.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            redacted = tuple(
                _SHARE_PATH.sub(r"\1<redacted>", item) if isinstance(item, str) else item
                for item in record.args
            )
            if redacted != record.args:
                record.args = redacted
        if isinstance(record.msg, str) and "/share/" in record.msg:
            record.msg = _SHARE_PATH.sub(r"\1<redacted>", record.msg)
        return True
