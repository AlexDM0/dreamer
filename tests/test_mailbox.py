import threading
import time

from server.mailbox import Mailbox


def test_take_returns_put_item():
    mb = Mailbox()
    mb.put("a")
    assert mb.take(timeout=0) == "a"
    assert mb.take(timeout=0) is None


def test_put_overwrites_latest_wins():
    mb = Mailbox()
    for i in range(10):
        mb.put(i)
    assert mb.take(timeout=0) == 9
    assert mb.take(timeout=0.01) is None


def test_take_timeout_returns_none():
    mb = Mailbox()
    started = time.perf_counter()
    assert mb.take(timeout=0.05) is None
    assert time.perf_counter() - started >= 0.04


def test_clear_empties_slot():
    mb = Mailbox()
    mb.put("x")
    mb.clear()
    assert mb.take(timeout=0) is None


def test_none_item_is_not_confused_with_empty():
    mb = Mailbox()
    mb.put(None)
    mb.put("real")
    assert mb.take(timeout=0) == "real"


def test_take_wakes_on_put_from_other_thread():
    mb = Mailbox()
    got = []
    t = threading.Thread(target=lambda: got.append(mb.take(timeout=2)))
    t.start()
    time.sleep(0.05)
    mb.put("hello")
    t.join(1)
    assert got == ["hello"]


def test_wake_ends_a_waiting_take_without_an_item():
    mb = Mailbox()
    got = []
    waiter = threading.Thread(target=lambda: got.append(mb.take(timeout=5)))
    waiter.start()
    mb.wake()
    waiter.join(1)
    assert got == [None] and not waiter.is_alive()
    mb.put("next")
    assert mb.take(timeout=0) == "next"  # one wake, not a permanent state
