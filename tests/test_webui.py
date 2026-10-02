from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from breeze_infer import webui
from breeze_infer.api import app
from breeze_infer.profiles import ProfileStore


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app.state, "cfg", SimpleNamespace(max_ref_audio_bytes=1024), raising=False)
    return TestClient(app)


def mock_stt(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        webui,
        "_stt_client",
        lambda: httpx.AsyncClient(transport=transport),
    )


def test_webui_routes_and_config_do_not_expose_key(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://stt:5092/v1")
    monkeypatch.setenv("BREEZE_STT_API_KEY", "private-key")

    assert "Breeze TTS" in client.get("/").text
    assert "buildSpeechPayload" in client.get("/webui.mjs").text
    assert client.get("/webui.css").status_code == 200
    assert client.get("/webui/config").json() == {"stt_base_url": "http://stt:5092/v1"}
    assert "private-key" not in client.get("/webui/config").text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://stt:5092", "http://stt:5092/v1"),
        ("https://example.test/prefix/v1/", "https://example.test/prefix/v1"),
    ],
)
def test_normalize_stt_url(value, expected):
    assert webui.normalize_stt_base_url(value) == expected


@pytest.mark.parametrize("value", ["", "ftp://stt", "http://user:secret@stt", "http://stt/?q=1"])
def test_invalid_stt_url(value):
    with pytest.raises(ValueError):
        webui.normalize_stt_base_url(value)


def test_transcribe_discovers_model_and_uses_configured_key(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://stt:5092")
    monkeypatch.setenv("BREEZE_STT_API_KEY", "private-key")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers.get("authorization") == "Bearer private-key"
        if request.method == "GET":
            assert str(request.url) == "http://stt:5092/v1/models"
            return httpx.Response(200, json={"data": [{"id": "parakeet-unified-en-0.6b"}]})
        assert str(request.url) == "http://stt:5092/v1/audio/transcriptions"
        assert b"parakeet-unified-en-0.6b" in request.content
        assert b"sample audio" in request.content
        return httpx.Response(200, json={"text": "spoken reference"})

    mock_stt(monkeypatch, handler)
    response = client.post("/webui/transcribe", files={"file": ("sample.wav", b"sample audio", "audio/wav")})

    assert response.status_code == 200
    assert response.json() == {"text": "spoken reference", "model": "parakeet-unified-en-0.6b"}
    assert len(requests) == 2


def test_changed_stt_url_does_not_receive_configured_key(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://default:5092/v1")
    monkeypatch.setenv("BREEZE_STT_API_KEY", "private-key")

    def handler(request):
        assert "authorization" not in request.headers
        if request.method == "GET":
            return httpx.Response(503)
        assert b"whisper-1" in request.content
        return httpx.Response(200, json={"text": "fallback transcript"})

    mock_stt(monkeypatch, handler)
    response = client.post(
        "/webui/transcribe",
        data={"stt_base_url": "http://other:8080"},
        files={"file": ("clip.wav", b"audio", "audio/wav")},
    )
    assert response.json() == {"text": "fallback transcript", "model": "whisper-1"}


def test_session_key_overrides_configured_key(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://stt:5092")
    monkeypatch.setenv("BREEZE_STT_API_KEY", "default-key")

    def handler(request):
        assert request.headers.get("authorization") == "Bearer session-key"
        return httpx.Response(200, json={"data": [{"id": "parakeet"}]}) if request.method == "GET" else httpx.Response(200, json={"text": "hello"})

    mock_stt(monkeypatch, handler)
    response = client.post(
        "/webui/transcribe",
        data={"api_key": "session-key"},
        files={"file": ("clip.wav", b"audio", "audio/wav")},
    )
    assert response.status_code == 200


def test_transcription_input_limit_and_upstream_failure(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://stt:5092")
    app.state.cfg.max_ref_audio_bytes = 4
    too_large = client.post("/webui/transcribe", files={"file": ("clip.wav", b"12345", "audio/wav")})
    assert too_large.status_code == 413

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "parakeet"}]}) if request.method == "GET" else httpx.Response(401, text="unauthorized")

    mock_stt(monkeypatch, handler)
    failed = client.post("/webui/transcribe", files={"file": ("clip.wav", b"1234", "audio/wav")})
    assert failed.status_code == 502
    assert "HTTP 401" in failed.json()["detail"]


def test_transcription_rejects_empty_text_and_missing_file(client, monkeypatch):
    monkeypatch.setenv("BREEZE_STT_BASE_URL", "http://stt:5092")
    assert client.post("/webui/transcribe").status_code == 422

    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "parakeet"}]}) if request.method == "GET" else httpx.Response(200, json={"text": "  "})

    mock_stt(monkeypatch, handler)
    response = client.post("/webui/transcribe", files={"file": ("clip.wav", b"audio", "audio/wav")})
    assert response.status_code == 502
    assert "no transcript" in response.json()["detail"]


def test_voice_upload_edit_and_delete_via_webui_endpoints(client, monkeypatch, tmp_path):
    monkeypatch.setattr(app.state, "profiles", ProfileStore(tmp_path), raising=False)
    created = client.post(
        "/upload_voice",
        data={"name": "sample", "ref_text": "hello there", "preload": "false"},
        files={"voice_file": ("clip.wav", b"fake audio", "audio/wav")},
    )
    assert created.status_code == 200
    voice_id = created.json()["voice_id"]
    assert voice_id == "sample"
    assert any(item["id"] == voice_id for item in client.get("/v1/audio/voices").json()["data"])

    updated = client.patch(
        f"/v1/audio/voices/{voice_id}",
        json={"name": "Renamed", "ref_text": "new transcript"},
    )
    assert updated.status_code == 200
    assert updated.json()["ref_text"] == "new transcript"
    assert client.delete(f"/v1/audio/voices/{voice_id}").status_code == 200
    assert not any(item["id"] == voice_id for item in client.get("/v1/audio/voices").json()["data"])
