"""Built-in browser UI and OpenAI-compatible transcription relay."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile

router = APIRouter()
ASSETS = Path(__file__).with_name("webui_assets")
FALLBACK_STT_MODEL = "whisper-1"


def normalize_stt_base_url(value: str) -> str:
    """Accept an HTTP(S) server root or a URL ending in /v1."""
    value = value.strip().rstrip("/")
    try:
        parts = urlsplit(value)
        # Accessing port also checks for malformed port numbers.
        _ = parts.port
    except ValueError as exc:
        raise ValueError("Enter a valid STT server URL.") from exc
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or any(char.isspace() for char in value)
    ):
        raise ValueError("Enter an HTTP or HTTPS STT server URL without credentials or a query.")
    return value if parts.path.endswith("/v1") else f"{value}/v1"


def _default_stt_url() -> str:
    return os.getenv("BREEZE_STT_BASE_URL", "").strip()


def _stt_key(selected_url: str, session_key: str) -> str:
    if session_key:
        return session_key
    configured = _default_stt_url()
    if not configured:
        return ""
    try:
        if normalize_stt_base_url(configured) == selected_url:
            return os.getenv("BREEZE_STT_API_KEY", "").strip()
    except ValueError:
        pass
    return ""


def _stt_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=300, write=60, pool=10))


async def _discover_stt_model(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> str:
    try:
        response = await client.get(f"{url}/models", headers=headers, timeout=10)
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("data", []):
            if isinstance(item, dict) and item.get("id"):
                return str(item["id"])
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        pass
    return FALLBACK_STT_MODEL


@router.get("/", include_in_schema=False)
def webui_page() -> FileResponse:
    return FileResponse(ASSETS / "index.html", media_type="text/html")


@router.get("/webui.css", include_in_schema=False)
def webui_css() -> FileResponse:
    return FileResponse(ASSETS / "webui.css", media_type="text/css")


@router.get("/webui.mjs", include_in_schema=False)
def webui_script() -> FileResponse:
    return FileResponse(ASSETS / "webui.mjs", media_type="text/javascript")


@router.get("/webui/config")
def webui_config() -> dict[str, str]:
    # API keys never appear in the page or its configuration payload.
    return {"stt_base_url": _default_stt_url()}


@router.post("/webui/transcribe")
async def webui_transcribe(request: Request) -> dict[str, str]:
    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, UploadFile) or not upload.filename:
        raise HTTPException(422, "Choose a reference audio file to transcribe.")
    limit = request.app.state.cfg.max_ref_audio_bytes
    audio = await upload.read(limit + 1)
    if not audio:
        raise HTTPException(422, "Reference audio is empty.")
    if len(audio) > limit:
        raise HTTPException(413, "Reference audio exceeds the configured size limit.")

    requested_url = str(form.get("stt_base_url") or _default_stt_url())
    try:
        base_url = normalize_stt_base_url(requested_url)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    key = _stt_key(base_url, str(form.get("api_key") or "").strip())
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with _stt_client() as client:
            model = await _discover_stt_model(client, base_url, headers)
            response = await client.post(
                f"{base_url}/audio/transcriptions",
                headers=headers,
                data={"model": model},
                files={
                    "file": (
                        Path(upload.filename).name,
                        audio,
                        upload.content_type or "application/octet-stream",
                    )
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "The STT server timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"STT server returned HTTP {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach the STT server: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(502, "STT server returned invalid JSON.") from exc

    transcript = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
    if not transcript:
        raise HTTPException(502, "STT server returned no transcript text.")
    return {"text": transcript, "model": model}
