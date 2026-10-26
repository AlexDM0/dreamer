"""Start-up guards in server/__main__.py."""
import logging
import socket

from server import __main__ as entry


def test_port_in_use_is_detected():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert entry._port_in_use("127.0.0.1", port)
    assert not entry._port_in_use("127.0.0.1", port)


def test_log_file_is_written(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        entry._log_to_file(str(tmp_path))
        logging.getLogger("dreamer").warning("hello log file")
        for handler in root.handlers:
            handler.flush()
        assert "hello log file" in (tmp_path / entry.LOG_FILE).read_text(encoding="utf-8")
    finally:
        for handler in root.handlers[len(before):]:
            root.removeHandler(handler)
            handler.close()
        logging.captureWarnings(False)
