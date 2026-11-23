from contextlib import contextmanager

import httpx
import pytest

import mouseslip
import mouseslip.predict


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self):
        yield self.body


def fake_stream(body: bytes):
    @contextmanager
    def stream(*args, **kwargs):
        yield FakeResponse(body)

    return stream


def test_fetch_model_refuses_a_download_that_does_not_match_the_checksum(tmp_path, monkeypatch):
    target = tmp_path / "tslm.pt"
    monkeypatch.setattr(mouseslip.predict, "TSLM_CHECKPOINT", target)
    monkeypatch.setattr(httpx, "stream", fake_stream(b"not the model"))
    with pytest.raises(SystemExit):
        mouseslip.fetch_model()
    assert not target.exists()
    assert not target.with_suffix(".part").exists()


def test_fetch_model_saves_a_matching_download(tmp_path, monkeypatch):
    target = tmp_path / "tslm.pt"
    monkeypatch.setattr(mouseslip.predict, "TSLM_CHECKPOINT", target)
    monkeypatch.setattr(httpx, "stream", fake_stream(b"weights"))
    monkeypatch.setattr(mouseslip, "MODEL_SHA256", __import__("hashlib").sha256(b"weights").hexdigest())
    mouseslip.fetch_model()
    assert target.read_bytes() == b"weights"
