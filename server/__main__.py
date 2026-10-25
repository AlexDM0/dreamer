"""``python -m server``: load config and run Dreamer under uvicorn."""
from __future__ import annotations

import logging
import socket
import sys

import uvicorn

from . import protocol
from .app import create_app
from .config import load_config


def _lan_address() -> str | None:
    """Best-effort LAN IP for the startup banner (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        config = load_config()
    except ValueError as exc:
        logging.getLogger("dreamer").error("%s", exc)
        return 2
    app = create_app(config)
    ip = _lan_address()
    if ip:
        logging.getLogger("dreamer").info(
            "Dreamer: kid page http://%s:%d/  admin http://%s:%d/admin", ip, config.port, ip, config.port)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        # Header + 2 MB sketch; larger messages are rejected by the WebSocket layer.
        ws_max_size=protocol.HEADER_SIZE + protocol.MAX_MESSAGE_BYTES,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
