"""Admin authentication: PIN check with lockout, and an HMAC-signed session cookie.

Cookie value: ``<expiry unix seconds>.<hex HMAC-SHA256(key, expiry)>``, where ``key`` is
derived from the secret and the PIN, so changing the PIN logs every device out.

Lockout: after MAX_FAILURES wrong PINs a client is locked out for LOCKOUT_SECONDS. Until
it logs in successfully, every further wrong PIN locks it out again for twice as long
(capped at LOCKOUT_MAX_SECONDS). A global budget does the same across all clients, so
using many addresses does not multiply the guessing rate.
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
LOCKOUT_MAX_SECONDS = 15 * 60.0
GLOBAL_KEY = "*"               # the all-clients budget
GLOBAL_MAX_FAILURES = 20
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
        # Cookie key bound to the PIN: a new PIN invalidates existing cookies.
        self._key = hmac.new(secret, b"dreamer-admin-cookie\0" + self._pin, hashlib.sha256).digest()
        self._clock = clock
        self._lock = threading.Lock()
        # key (client or GLOBAL_KEY) -> (failures since last success, locked-until timestamp)
        self._failures: dict[str, tuple[int, float]] = {}

    def check_pin(self, pin: str, client: str = "") -> bool:
        """True if correct. Raises LockedOut while the client (or everyone) is locked out."""
        now = self._clock()
        with self._lock:
            until = max(self._failures.get(k, (0, 0.0))[1] for k in (client, GLOBAL_KEY))
            if until > now:
                raise LockedOut(until - now)
            if hmac.compare_digest(pin.encode("utf-8"), self._pin):
                self._failures.pop(client, None)
                self._failures.pop(GLOBAL_KEY, None)
                return True
            self._fail(client, MAX_FAILURES, now)
            self._fail(GLOBAL_KEY, GLOBAL_MAX_FAILURES, now)
            return False

    def _fail(self, key: str, limit: int, now: float) -> None:
        count = self._failures.get(key, (0, 0.0))[0] + 1
        until = 0.0
        if count >= limit:
            # limit -> 30 s, limit+1 -> 60 s, ... capped; only a correct PIN resets it.
            seconds = min(LOCKOUT_SECONDS * 2 ** min(count - limit, 16), LOCKOUT_MAX_SECONDS)
            until = now + seconds
            log.warning("admin login: %d wrong PINs from %s; locked out for %.0fs",
                        count, "all clients" if key == GLOBAL_KEY else key, seconds)
        self._failures[key] = (count, until)

    def _sign(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("ascii"), hashlib.sha256).hexdigest()

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
