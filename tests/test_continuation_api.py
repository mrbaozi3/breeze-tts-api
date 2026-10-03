from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from starlette.requests import ClientDisconnect

import breeze_infer.api as api_module
from breeze_infer.api import app, continuation_speech, speech
from breeze_infer.continuation import ContinuationManager
from breeze_infer.observability import new_service_metrics


class _Request:
    def __init__(self, body: dict):
        self.body = body
        self.headers = {"content-type": "application/json"}

    async def json(self):
        return self.body

    async def is_disconnected(self) -> bool:
        return False


class _DisconnectRequest(_Request):
    disconnected = False

    async def is_disconnected(self) -> bool:
        return self.disconnected


class _FormRequest(_Request):
    def __init__(self, body: dict):
        super().__init__(body)
        self.headers = {"content-type": "multipart/form-data"}

    async def form(self):
        return self.body


class _Profiles:
    def resolve(self, _identifier):
        from breeze_infer.profiles import ProfileNotFoundError

        raise ProfileNotFoundError

    def list(self):
        return []


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(max(1, len(text.split()))))}

    def convert_ids_to_tokens(self, ids):
        return ["token"] * len(ids)


class _ContinuationRuntime:
    sample_rate = 24_000

    def __init__(self):
        self.closed = []

    def new_session(self, continuation_id, _inputs):
        return SimpleNamespace(
            continuation_id=continuation_id,
            chunk_index=0,
            last_timing={},
            last_used_at=time.monotonic(),
            closed=False,
            cache_length=8,
        )

    def preflight_continue(self, _state, text, **_kwargs):
        if text == "too long":
            from models.continuation_streaming import ContinuationContextError

            raise ContinuationContextError("continuation context limit exceeded")
        return {"prepared": True}

    def _chunks(self, state, operation):
        yield SimpleNamespace(audio=np.array([0.25, -0.25], dtype=np.float32))
        state.last_timing = {
            "text_encoder_ms": 1.0 if operation == "continue" else 0.0,
            "append_prefill_ms": 2.0,
            "cache_length_after": 20,
        }
        state.chunk_index += 1

    def iter_start(self, state, _inputs, **_kwargs):
        return self._chunks(state, "start")

    def iter_continue(self, state, _text, **_kwargs):
        return self._chunks(state, "continue")

    def close(self, state):
        state.closed = True
        self.closed.append(state.continuation_id)


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    continuation_runtime = _ContinuationRuntime()

    class _NormalRuntime:
        fast_enabled = True
        sample_rate = 24_000

        def iter_audio_chunks(self, _inputs, **_kwargs):
            yield SimpleNamespace(audio=np.array([0.1, -0.1], dtype=np.float32))

    app.state.metrics = new_service_metrics()
    app.state.profiles = _Profiles()
    app.state.runtime = _NormalRuntime()
    app.state.tokenizer = _Tokenizer()
    app.state.audio_tokenizer = object()
    app.state.model = object()
    app.state.continuations = ContinuationManager(continuation_runtime)
    app.state.cfg = SimpleNamespace(
        max_ref_audio_bytes=1024,
        weights=None,
        fast_all=None,
        fast_text_encoder=False,
        fast_backbone_prefill=False,
        fast_backbone_decode=True,
        fast_depth_decoder=False,
        fast_codec=False,
        continuation_segment_gap_ms=10.0,
    )
    app.state.start_time = time.monotonic()
    monkeypatch.setattr(api_module, "set_all_seeds", lambda _seed: None)
    monkeypatch.setattr(
        api_module,
        "prepare_inputs",
        lambda *_args, **_kwargs: {
            "input_ids": torch.zeros((1, 8), dtype=torch.long),
            "cfg_negative_prompt_ids": torch.zeros((1, 8), dtype=torch.long),
        },
    )
    yield continuation_runtime
    if api_module._request_lock.locked():
        api_module._request_lock.release()


def _payload(continuation_id="one", **overrides):
    result = {
        "input": "hello there",
        "continuation_id": continuation_id,
        "guidance_scale": 1,
        "seed": 42,
        "response_format": "pcm",
    }
    result.update(overrides)
    return result


def test_id_implicitly_starts_then_continues(configured) -> None:
    first = asyncio.run(continuation_speech(_Request(_payload())))
    second = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    assert first.headers["x-continuation-chunk-index"] == "0"
    assert second.headers["x-continuation-chunk-index"] == "1"
    assert first.headers["x-continuation-restarted"] == "false"
    assert app.state.continuations.session.successful_chunks == 2
    assert configured.closed == []


def test_standard_speech_endpoint_dispatches_continuation_id(configured) -> None:
    first = asyncio.run(speech(_Request(_payload())))
    second = asyncio.run(speech(_Request(_payload(input="again"))))

    assert first.headers["x-continuation-chunk-index"] == "0"
    assert second.headers["x-continuation-chunk-index"] == "1"
    assert app.state.continuations.session.successful_chunks == 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="ffmpeg and ffprobe are required")
@pytest.mark.parametrize(
    ("audio_format", "media_type", "codec", "output_rate"),
    [
        ("mp3", "audio/mpeg", "mp3", 24_000),
        ("opus", "audio/ogg", "opus", 48_000),
        ("aac", "audio/aac", "aac", 24_000),
    ],
)
def test_continuation_returns_decodable_compressed_audio(
    configured, monkeypatch, audio_format, media_type, codec, output_rate
) -> None:
    monkeypatch.setattr(api_module, "_pcm16", lambda _audio: bytes(2400 * 2))

    first = asyncio.run(continuation_speech(_Request(_payload(response_format=audio_format))))
    second = asyncio.run(continuation_speech(
        _Request(_payload(input="again", response_format=audio_format))
    ))
    for response in (first, second):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
             "-of", "json", "-i", "pipe:0"],
            input=response.body, capture_output=True, check=True,
        )
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "pipe:1"],
            input=response.body, capture_output=True, check=True,
        )
        assert response.media_type == media_type
        assert response.headers["x-sample-rate"] == str(output_rate)
        assert response.headers["x-audio-sample-rate"] == str(output_rate)
        assert json.loads(probe.stdout)["streams"][0] == {
            "codec_name": codec, "sample_rate": str(output_rate), "channels": 1,
        }
        assert decoded.stdout
    assert second.headers["x-continuation-chunk-index"] == "1"
    assert app.state.continuations.session.successful_chunks == 2


def test_multipart_continuation_accepts_compressed_format() -> None:
    response = asyncio.run(continuation_speech(
        _FormRequest(_payload(response_format="mp3"))
    ))
    assert response.media_type == "audio/mpeg"
    assert response.body


def test_continuation_keeps_default_wav() -> None:
    response = asyncio.run(continuation_speech(_Request(_payload(response_format=None))))
    assert response.media_type == "audio/wav"
    assert response.body.startswith(b"RIFF")


def test_invalid_continuation_format_returns_422() -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(continuation_speech(_Request(_payload(response_format="flac"))))
    assert error.value.status_code == 422
    assert app.state.continuations.session is None


@pytest.mark.parametrize("audio_format", ["mp3", "opus", "aac"])
@pytest.mark.parametrize("stream_fields", [{"stream": True}, {"stream_format": "sse"}])
def test_continuation_streaming_rejects_compressed_formats(audio_format, stream_fields) -> None:
    with pytest.raises(HTTPException) as error:
        asyncio.run(continuation_speech(_Request(_payload(response_format=audio_format,
                                                            **stream_fields))))
    assert error.value.status_code == 422
    assert app.state.continuations.session is None


def test_continuation_encoder_failure_closes_session(configured, monkeypatch) -> None:
    asyncio.run(continuation_speech(_Request(_payload())))
    monkeypatch.setattr(
        api_module.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout=b"", stderr=b"encoder failed"),
    )
    with pytest.raises(HTTPException, match="mp3 encoding failed") as error:
        asyncio.run(continuation_speech(_Request(_payload(input="again", response_format="mp3"))))
    assert error.value.status_code == 500
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None
    assert not api_module._request_lock.locked()


def test_different_id_replaces_idle_session(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload("one"))))
    response = asyncio.run(continuation_speech(_Request(_payload("two"))))

    assert configured.closed == ["one"]
    assert response.headers["x-continuation-restarted"] == "true"
    assert app.state.continuations.session.continuation_id == "two"


def test_same_id_mismatch_is_rejected_without_destroying_state() -> None:
    asyncio.run(continuation_speech(_Request(_payload())))

    with pytest.raises(HTTPException, match="instruction") as exc_info:
        asyncio.run(
            continuation_speech(_Request(_payload(instructions="speak warmly")))
        )

    assert exc_info.value.status_code == 409
    assert app.state.continuations.session.continuation_id == "one"


def test_normal_speech_cleans_idle_continuation(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload())))
    response = asyncio.run(speech(_Request({"input": "normal", "response_format": "pcm"})))

    assert response.status_code == 200
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None


def test_continuation_requires_client_id() -> None:
    with pytest.raises(HTTPException, match="continuation_id") as exc_info:
        asyncio.run(continuation_speech(_Request({"input": "hello"})))

    assert exc_info.value.status_code == 422


@pytest.mark.parametrize("text", [".", "...", "!? —"])
def test_continuation_rejects_punctuation_only_text(text: str) -> None:
    with pytest.raises(HTTPException, match="letter or digit") as exc_info:
        asyncio.run(continuation_speech(_Request(_payload(input=text))))

    assert exc_info.value.status_code == 422
    assert app.state.continuations.session is None


def test_omitted_and_explicit_default_cfg_match() -> None:
    asyncio.run(
        continuation_speech(
            _Request(
                {
                    "input": "hello",
                    "continuation_id": "one",
                    "response_format": "pcm",
                }
            )
        )
    )
    response = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    assert response.headers["x-continuation-chunk-index"] == "1"


def test_stream_releases_request_lock_but_retains_session() -> None:
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )
    assert not api_module._request_lock.locked()

    async def consume() -> bytes:
        parts = []
        async for part in response.body_iterator:
            parts.append(part)
        return b"".join(parts)

    assert asyncio.run(consume())
    asyncio.run(response.background())

    assert not api_module._request_lock.locked()
    assert app.state.continuations.session is not None
    assert app.state.continuations.session.in_flight is False


def test_asgi_disconnect_before_continuation_body_closes_session(configured) -> None:
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )
    assert app.state.continuations.session.in_flight is True

    async def exercise() -> None:
        async def receive():
            return {"type": "http.disconnect"}

        async def send(_message):
            await asyncio.Event().wait()

        await response(
            {"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send
        )

    asyncio.run(exercise())
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None
    assert not api_module._request_lock.locked()
    assert asyncio.run(continuation_speech(_Request(_payload()))).body


def test_asgi_continuation_send_error_closes_session(configured) -> None:
    iterator_closed = threading.Event()

    def chunks(_state, _inputs, **_kwargs):
        try:
            yield SimpleNamespace(audio=np.array([0.25], dtype=np.float32))
            yield SimpleNamespace(audio=np.array([0.25], dtype=np.float32))
        finally:
            iterator_closed.set()

    configured.iter_start = chunks
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )

    async def exercise() -> None:
        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("client disconnected")

        with pytest.raises(ClientDisconnect):
            await response(
                {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
            )

    asyncio.run(exercise())
    assert iterator_closed.is_set()
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None
    assert not api_module._request_lock.locked()


def test_asgi_continuation_completion_retains_session(configured) -> None:
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )
    sent = []

    async def exercise() -> None:
        async def receive():
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)

        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send
        )

    asyncio.run(exercise())
    assert any(message.get("body") for message in sent)
    assert configured.closed == []
    assert app.state.continuations.session.successful_chunks == 1
    assert app.state.continuations.session.in_flight is False
    assert not api_module._request_lock.locked()


def test_asgi_continuation_disconnect_waits_for_model_step(configured) -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow_chunks(state, _inputs, **_kwargs):
        yield SimpleNamespace(audio=np.array([0.25], dtype=np.float32))
        entered.set()
        release.wait(2)
        state.chunk_index += 1

    configured.iter_start = slow_chunks
    response = asyncio.run(
        continuation_speech(_Request(_payload(stream=True, response_format="pcm")))
    )

    async def exercise() -> None:
        async def receive():
            await asyncio.to_thread(entered.wait, 2)
            return {"type": "http.disconnect"}

        async def send(_message):
            pass

        task = asyncio.create_task(
            response(
                {"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            await asyncio.sleep(0.05)
            assert not task.done()
            assert api_module._request_lock.locked()
        finally:
            release.set()
        await asyncio.wait_for(task, 2)

    asyncio.run(exercise())
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None
    assert not api_module._request_lock.locked()


def test_unstarted_stream_does_not_hold_lock_and_expires(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload(stream=True))))
    pending = app.state.continuations.session

    assert pending.in_flight is True
    assert pending.stream_started is False
    assert not api_module._request_lock.locked()
    pending.in_flight_since = 0.0
    assert app.state.continuations.expire(now=10.0) is True
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None


def test_normal_request_rejects_pending_continuation_without_holding_lock() -> None:
    asyncio.run(continuation_speech(_Request(_payload(stream=True))))

    with pytest.raises(HTTPException, match="continuation request") as exc_info:
        asyncio.run(speech(_Request({"input": "normal", "response_format": "pcm"})))

    assert exc_info.value.status_code == 409
    assert not api_module._request_lock.locked()


def test_failure_cleanup_error_still_releases_request_lock(configured) -> None:
    def fail_generation(*_args, **_kwargs):
        raise RuntimeError("generation failed")

    def fail_close(_state):
        raise RuntimeError("codec close failed")

    configured.iter_start = fail_generation
    configured.close = fail_close

    with pytest.raises(HTTPException, match="Continuation synthesis failed"):
        asyncio.run(continuation_speech(_Request(_payload())))

    assert not api_module._request_lock.locked()
    assert app.state.continuations.session is None


def test_stream_disconnect_after_audio_invalidates_session(configured) -> None:
    request = _DisconnectRequest(_payload(stream=True, response_format="pcm"))
    response = asyncio.run(continuation_speech(request))

    async def consume_until_disconnect() -> bytes:
        first = await anext(response.body_iterator)
        request.disconnected = True
        await asyncio.sleep(0.06)
        with pytest.raises(StopAsyncIteration):
            await anext(response.body_iterator)
        return first

    assert asyncio.run(consume_until_disconnect())
    asyncio.run(response.background())

    assert configured.closed == ["one"]
    assert app.state.continuations.session is None
    assert app.state.metrics["continuation"]["sessions_cancelled"] == 1
    assert app.state.metrics["continuation"]["sessions_completed"] == 0
    assert not api_module._request_lock.locked()


def test_stream_followup_context_error_is_409_before_response(configured) -> None:
    asyncio.run(continuation_speech(_Request(_payload())))

    with pytest.raises(HTTPException, match="context limit") as exc_info:
        asyncio.run(
            continuation_speech(_Request(_payload(input="too long", stream=True)))
        )

    assert exc_info.value.status_code == 409
    assert configured.closed == ["one"]
    assert app.state.continuations.session is None


def test_buffered_followup_prepends_configured_silence() -> None:
    first = asyncio.run(continuation_speech(_Request(_payload())))
    second = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    gap = bytes(24_000 * 10 // 1000 * 2)
    assert second.body.startswith(gap)
    assert second.body[len(gap) :] == first.body
    assert second.headers["x-continuation-gap-ms"] == "10.0"


def test_continuation_gap_uses_continuation_codec_sample_rate(configured) -> None:
    # The retained continuation runtime is authoritative for generated PCM;
    # the general runtime rate must not affect the inserted silence length.
    app.state.runtime.sample_rate = 48_000
    asyncio.run(continuation_speech(_Request(_payload())))
    second = asyncio.run(continuation_speech(_Request(_payload(input="again"))))

    expected_gap = bytes(24_000 * 10 // 1000 * 2)
    assert second.body.startswith(expected_gap)
    assert second.headers["x-sample-rate"] == "24000"


def test_raw_streaming_followup_prefixes_silence_to_first_audio_buffer() -> None:
    asyncio.run(continuation_speech(_Request(_payload())))
    response = asyncio.run(
        continuation_speech(_Request(_payload(input="again", stream=True)))
    )

    async def consume() -> list[bytes]:
        return [part async for part in response.body_iterator]

    parts = asyncio.run(consume())
    asyncio.run(response.background())
    expected_audio = api_module._pcm16(np.array([0.25, -0.25], dtype=np.float32))
    assert len(parts) == 1
    assert parts[0] == bytes(24_000 * 10 // 1000 * 2) + expected_audio


def test_sse_followup_prefixes_silence_to_first_audio_event() -> None:
    asyncio.run(continuation_speech(_Request(_payload())))
    response = asyncio.run(
        continuation_speech(
            _Request(_payload(input="again", stream=True, stream_format="sse"))
        )
    )

    async def consume() -> bytes:
        return b"".join([part async for part in response.body_iterator])

    body = asyncio.run(consume())
    asyncio.run(response.background())
    first_event = json.loads(body.splitlines()[0].removeprefix(b"data: "))
    assert first_event["type"] == "audio.chunk"
    expected_audio = api_module._pcm16(np.array([0.25, -0.25], dtype=np.float32))
    assert base64.b64decode(first_event["data"]) == bytes(
        24_000 * 10 // 1000 * 2
    ) + expected_audio
