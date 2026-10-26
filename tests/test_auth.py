import pytest
from conftest import ADMIN_PIN

from server.auth import (
    COOKIE_MAX_AGE,
    GLOBAL_MAX_FAILURES,
    GLOBAL_WINDOW_SECONDS,
    LOCKOUT_MAX_SECONDS,
    LOCKOUT_SECONDS,
    MAX_FAILURES,
    AdminAuth,
    LockedOut,
    load_or_create_secret,
)


class Clock:
    def __init__(self):
        self.now = 1_700_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def auth(clock):
    return AdminAuth("2468", b"secret", clock=clock)


def test_pin_check(auth):
    assert auth.check_pin("2468")
    assert not auth.check_pin("0000")


def test_lockout_after_five_failures_then_recovers(auth, clock):
    for _ in range(MAX_FAILURES):
        assert not auth.check_pin("0000", "ip1")
    with pytest.raises(LockedOut):
        auth.check_pin("2468", "ip1")  # even the right PIN is refused while locked
    assert auth.check_pin("2468", "ip2")  # other clients unaffected
    clock.now += LOCKOUT_SECONDS + 0.1
    assert auth.check_pin("2468", "ip1")


def test_lockout_escalates_until_success(auth, clock):
    for _ in range(MAX_FAILURES):
        auth.check_pin("0000", "ip1")
    waits = []
    for _ in range(3):
        with pytest.raises(LockedOut) as exc:
            auth.check_pin("0000", "ip1")
        waits.append(exc.value.retry_after)
        clock.now += exc.value.retry_after + 0.1
        assert not auth.check_pin("0000", "ip1")  # one guess, then locked again
    assert waits[1] == pytest.approx(2 * waits[0]) and waits[2] == pytest.approx(4 * waits[0])
    for _ in range(30):  # the lockout is capped
        clock.now += LOCKOUT_MAX_SECONDS + 1
        auth.check_pin("0000", "ip1")
    with pytest.raises(LockedOut) as exc:
        auth.check_pin("2468", "ip1")
    assert exc.value.retry_after <= LOCKOUT_MAX_SECONDS
    clock.now += LOCKOUT_MAX_SECONDS + 1
    assert auth.check_pin("2468", "ip1")
    assert not auth.check_pin("0000", "ip1")  # success reset the count: no lockout


def test_global_backstop_holds_only_clients_that_guessed(auth, clock):
    for i in range(GLOBAL_MAX_FAILURES):
        auth.check_pin("0000", f"ip{i}")  # one guess per address
    with pytest.raises(LockedOut):
        auth.check_pin("2468", "ip0")  # guessed before: held by the backstop
    assert auth.check_pin("2468", "parent-phone")  # no failures of its own: let through
    assert not auth.check_pin("0000", "fresh-ip")  # one guess per fresh address...
    with pytest.raises(LockedOut):
        auth.check_pin("2468", "fresh-ip")  # ...then held too
    clock.now += GLOBAL_WINDOW_SECONDS + 1
    assert auth.check_pin("2468", "ip0")


def test_wrong_pins_on_the_ipad_gate_do_not_lock_the_parent_out(auth, clock):
    wrong = 0
    while wrong < 20:  # the old global limit; waiting out each lockout in between
        try:
            assert not auth.check_pin("0000", "ipad")
            wrong += 1
        except LockedOut as exc:
            clock.now += exc.retry_after + 0.1
    with pytest.raises(LockedOut):
        auth.check_pin("2468", "ipad")
    assert auth.check_pin("2468", "parent-phone")


def test_pin_change_invalidates_cookies(clock):
    cookie = AdminAuth("2468", b"secret", clock=clock).make_cookie()
    assert AdminAuth("2468", b"secret", clock=clock).verify_cookie(cookie)
    assert not AdminAuth("1357", b"secret", clock=clock).verify_cookie(cookie)


def test_success_resets_failure_count(auth):
    for _ in range(MAX_FAILURES - 1):
        auth.check_pin("0000")
    assert auth.check_pin("2468")
    for _ in range(MAX_FAILURES - 1):
        assert not auth.check_pin("0000")  # no lockout yet


def test_cookie_roundtrip_and_expiry(auth, clock):
    cookie = auth.make_cookie()
    assert auth.verify_cookie(cookie)
    clock.now += COOKIE_MAX_AGE + 1
    assert not auth.verify_cookie(cookie)


@pytest.mark.parametrize("bad", [None, "", "abc", "123.", "9999999999.deadbeef", "١٢٣.ff", "x.y.z"])
def test_cookie_rejects_garbage(auth, bad):
    assert not auth.verify_cookie(bad)


def test_cookie_signed_with_other_secret_rejected(clock):
    other = AdminAuth("2468", b"other", clock=clock).make_cookie()
    assert not AdminAuth("2468", b"secret", clock=clock).verify_cookie(other)


def test_secret_persisted(tmp_path):
    a = load_or_create_secret("", tmp_path)
    assert (tmp_path / "secret").is_file()
    assert load_or_create_secret("", tmp_path) == a
    assert load_or_create_secret("fixed", tmp_path) == b"fixed"


# ----- HTTP -------------------------------------------------------------------------

PROTECTED = [
    ("get", "/api/admin/state", None),
    ("post", "/api/admin/styles", {"label": "x", "prompt": "y"}),
    ("patch", "/api/admin/items/styles/cartoon", {"hidden": True}),
    ("delete", "/api/admin/items/styles/cartoon", None),
    ("post", "/api/admin/order", {"kind": "topics", "ids": []}),
    ("post", "/api/admin/select", {"topic": "space"}),
]


@pytest.mark.parametrize("method,url,body", PROTECTED)
def test_protected_endpoints_401(client, method, url, body):
    kwargs = {"json": body} if body is not None else {}
    assert getattr(client, method)(url, **kwargs).status_code == 401


def test_upload_requires_login(client):
    r = client.post("/api/admin/style-image", data={"label": "x"},
                    files={"file": ("a.png", b"123", "image/png")})
    assert r.status_code == 401


def test_login_logout_flow(client):
    assert client.post("/api/admin/login", json={"pin": "0000"}).status_code == 401
    r = client.post("/api/admin/login", json={"pin": ADMIN_PIN})
    assert r.status_code == 200
    set_cookie = r.headers["set-cookie"].lower()
    assert "sg_admin=" in set_cookie and "httponly" in set_cookie
    assert client.get("/api/admin/state").status_code == 200
    assert client.post("/api/admin/logout").status_code == 200
    assert client.get("/api/admin/state").status_code == 401


def test_login_max_age_phone_vs_parent_gate(client):
    """The phone keeps a 30-day login; the iPad's parent gate asks for a 15-minute one."""
    long = client.post("/api/admin/login", json={"pin": ADMIN_PIN}).headers["set-cookie"]
    assert f"max-age={30 * 24 * 3600}" in long.lower()
    short = client.post("/api/admin/login", json={"pin": ADMIN_PIN, "short": True})
    assert short.status_code == 200
    assert "max-age=900" in short.headers["set-cookie"].lower()


def test_short_cookie_expires_after_15_minutes():
    now = [1000.0]
    auth = AdminAuth("2468", b"secret", clock=lambda: now[0])
    cookie = auth.make_cookie(15 * 60)
    now[0] += 15 * 60 - 1
    assert auth.verify_cookie(cookie)
    now[0] += 2
    assert not auth.verify_cookie(cookie)


def test_login_lockout_http(client):
    for _ in range(MAX_FAILURES):
        assert client.post("/api/admin/login", json={"pin": "0000"}).status_code == 401
    r = client.post("/api/admin/login", json={"pin": ADMIN_PIN})
    assert r.status_code == 429
    assert "retry-after" in r.headers


def test_forged_cookie_rejected(client):
    client.cookies.set("sg_admin", "9999999999.00")
    assert client.get("/api/admin/state").status_code == 401
