"""``python -m server``: load config and run Dreamer under uvicorn."""
from __future__ import annotations

import logging
import logging.handlers
import socket
import sys
from pathlib import Path

import uvicorn

from . import offline, protocol
from .config import load_config


def _lan_address() -> str | None:
    """Best-effort LAN IP for the startup banner (no packets are sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILE = "logs/dreamer.log"  # under data_dir; kept after the console window closes
ALREADY_RUNNING_EXIT_CODE = 3


def _log_to_file(data_dir: str) -> None:
    """Also log to data_dir/logs/dreamer.log (rotated, 3 x 5 MB)."""
    path = Path(data_dir) / LOG_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3,
                                                       encoding="utf-8")
    except OSError as exc:
        logging.getLogger("dreamer").warning("could not open the log file %s: %s", path, exc)
        return
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(handler)
    logging.captureWarnings(True)


def _port_in_use(host: str, port: int) -> bool:
    """True when another process (usually another Dreamer) already listens on the port.
    Checked before the models load: two processes loading onto the GPU at once can crash it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return True
    return False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        config = load_config()
    except ValueError as exc:
        logging.getLogger("dreamer").error("%s", exc)
        return 2
    _log_to_file(config.data_dir)
    if _port_in_use(config.host, config.port):
        logging.getLogger("dreamer").error(
            "Port %d is already in use: Dreamer is probably already running. Close the other "
            "Dreamer window first.", config.port)
        return ALREADY_RUNNING_EXIT_CODE
    offline.apply(config)  # before anything imports the model libraries
    from .app import create_app

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
