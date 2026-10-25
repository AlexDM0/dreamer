"""Admin authentication: PIN check with lockout, and an HMAC-signed session cookie.

Cookie value: ``<expiry unix seconds>.<hex HMAC-SHA256(secret, expiry)>``.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("dreamer.auth")

COOKIE_NAME = "sg_admin"
COOKIE_MAX_AGE = 30 * 24 * 3600
MAX_FAILURES = 5
LOCKOUT_SECONDS = 30.0
SECRET_FILE = "secret"


class LockedOut(Exception):
    """Too many wrong PINs; ``retry_after`` seconds remain."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"locked out for {retry_after:.0f}s")
        self.retry_after = retry_after


def load_or_create_secret(secret_key: str, data_dir: str | os.PathLike) -> bytes:
    """Return the cookie-signing key: config value, else ``data_dir/secret`` (created once)."""
    if secret_key:
        return secret_key.encode("utf-8")
    path = Path(data_dir) / SECRET_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing.encode("utf-8")
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)
    log.info("generated a new cookie secret at %s", path)
    return value.encode("utf-8")


class AdminAuth:
    """PIN verification (with in-memory per-client lockout) and cookie signing."""

    def __init__(self, pin: str, secret: bytes, clock: Callable[[], float] = time.time) -> None:
        self._pin = pin.encode("utf-8")
        self._secret = secret
        self._clock = clock
        self._lock = threading.Lock()
        # client key -> (consecutive failures, locked-until timestamp)
        self._failures: dict[str, tuple[int, float]] = {}

    def check_pin(self, pin: str, client: str = "") -> bool:
        """True if correct. Raises LockedOut while the client is locked out.

        After MAX_FAILURES wrong attempts the client is locked out for LOCKOUT_SECONDS;
        the counter then starts again.
        """
        now = self._clock()
        with self._lock:
            count, until = self._failures.get(client, (0, 0.0))
            if until > now:
                raise LockedOut(until - now)
            ok = hmac.compare_digest(pin.encode("utf-8"), self._pin)
            if ok:
                self._failures.pop(client, None)
                return True
            count += 1
            if count >= MAX_FAILURES:
                log.warning("admin login: %d wrong PINs from %s; locking out", count, client)
                self._failures[client] = (0, now + LOCKOUT_SECONDS)
            else:
                self._failures[client] = (count, 0.0)
            return False

    def _sign(self, payload: str) -> str:
        return hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).hexdigest()

    def make_cookie(self) -> str:
        expiry = str(int(self._clock()) + COOKIE_MAX_AGE)
        return f"{expiry}.{self._sign(expiry)}"

    def verify_cookie(self, value: str | None) -> bool:
        if not value or "." not in value:
            return False
        expiry, _, sig = value.partition(".")
        if not (expiry.isascii() and expiry.isdigit()):
            return False
        if not hmac.compare_digest(sig.encode("utf-8"), self._sign(expiry).encode("ascii")):
            return False
        return int(expiry) > self._clock()
