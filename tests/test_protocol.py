import pytest

from server import protocol


def test_pack_unpack_roundtrip():
    data = protocol.pack(protocol.SKETCH, 123456, b"payload")
    assert len(data) == protocol.HEADER_SIZE + 7
    assert data[0] == 0x01
    assert data[1:5] == (123456).to_bytes(4, "big")
    assert protocol.unpack(data) == (protocol.SKETCH, 123456, b"payload")


def test_picture_and_max_seq():
    data = protocol.pack_picture(protocol.MAX_SEQ, b"\xff\xd8")
    assert protocol.unpack(data) == (protocol.PICTURE, 0xFFFFFFFF, b"\xff\xd8")


def test_empty_payload_ok():
    assert protocol.unpack(protocol.pack(protocol.SKETCH, 0, b"")) == (protocol.SKETCH, 0, b"")


@pytest.mark.parametrize("bad", [b"", b"\x01", b"\x01\x00\x00\x00"])
def test_short_header_rejected(bad):
    with pytest.raises(protocol.ProtocolError):
        protocol.unpack(bad)


def test_unknown_type_rejected():
    with pytest.raises(protocol.ProtocolError):
        protocol.unpack(b"\x07\x00\x00\x00\x01abc")


@pytest.mark.parametrize("seq", [-1, 2**32])
def test_seq_range(seq):
    with pytest.raises(ValueError):
        protocol.pack(protocol.SKETCH, seq, b"")


def test_parse_client_text_strips_prompt_fields():
    msg = protocol.parse_client_text('{"t":"style","id":"crayon","prompt":"evil","text":"x"}')
    assert msg == {"t": "style", "id": "crayon"}


@pytest.mark.parametrize("text", ["not json", "[1,2]", '{"id":"x"}', '{"t":5}', "null"])
def test_parse_client_text_rejects_garbage(text):
    assert protocol.parse_client_text(text) is None
