"""File helpers shared by the stores that write atomically (gallery, styles)."""
from __future__ import annotations

import os
import time
from pathlib import Path

# Windows refuses to replace a file someone (antivirus, indexer, a reader) holds open;
# that clears within moments, so retry briefly.
REPLACE_RETRIES = 5


def replace_with_retry(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """``os.replace(src, dst)``, retried on PermissionError (up to ~0.75 s in total)."""
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(Path(src), Path(dst))
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
