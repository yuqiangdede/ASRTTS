from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.asr import AsrService
from app.config import load_config, project_root, resolve_path, save_config
from app.tts import MeloTtsError, MeloTtsService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = project_root()
CONFIG = load_config()
ASR_SERVICE = AsrService(CONFIG)
ASR_SERVICE_LOCK = threading.Lock()
TTS_SERVICE = MeloTtsService(CONFIG)
UPLOAD_DIR = resolve_path(CONFIG.get("storage", {}).get("upload_dir", "res/uploads"))
STATIC_DIR = ROOT / "app" / "web" / "static"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_FILES = 10

ASR_BACKEND_PRESETS = {
    "whisper": {
        "backend": "whisper",
        "model_path": "faster-whisper-large-v3-turbo",
    },
    "whisper_small": {
        "backend": "whisper",
        "model_path": "faster-whisper-small",
    },
    "funasr_nano": {
        "backend": "funasr_nano",
    },
}


def _read_index_html() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if suffix in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".aac", ".mp4"} else ".bin"


def _save_upload(upload: UploadFile) -> Path:
    suffix = _safe_suffix(upload.filename or "")
    target = UPLOAD_DIR / f"{int(time.time() * 1000)}_{Path(upload.filename or 'audio').stem}{suffix}"
    with target.open("wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    _prune_uploads(keep=MAX_UPLOAD_FILES)
    return target


def _prune_uploads(*, keep: int) -> None:
    files = [item for item in UPLOAD_DIR.iterdir() if item.is_file()]
    if len(files) <= keep:
        return

    files.sort(key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    for stale_file in files[keep:]:
        try:
            stale_file.unlink(missing_ok=True)
        except Exception:
            logger.warning("删除旧上传音频失败：%s", stale_file, exc_info=True)


def _get_asr_service() -> AsrService:
    with ASR_SERVICE_LOCK:
        return ASR_SERVICE


def _apply_asr_backend_config(selected_backend: str) -> dict:
    if selected_backend not in ASR_BACKEND_PRESETS:
        raise KeyError(selected_backend)

    updated = json.loads(json.dumps(CONFIG, ensure_ascii=False))
    asr_cfg = updated.setdefault("asr", {})
    preset = ASR_BACKEND_PRESETS[selected_backend]
    asr_cfg["backend"] = preset["backend"]
    if "model_path" in preset:
        asr_cfg["model_path"] = preset["model_path"]
    save_config(updated)
    return updated


def _reload_asr_service(config: dict) -> AsrService:
    global ASR_SERVICE, CONFIG

    new_service = AsrService(config)
    with ASR_SERVICE_LOCK:
        ASR_SERVICE = new_service
        CONFIG = config
        return ASR_SERVICE


def _current_backend_selection(asr_service: AsrService) -> str:
    if asr_service.backend == "funasr_nano":
        return "funasr_nano"
    model_name = Path(asr_service.model_path).name.lower()
    if "small" in model_name:
        return "whisper_small"
    return "whisper"


def create_app() -> FastAPI:
    app = FastAPI(title="ASR TTS Web", version="0.1.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_read_index_html(), headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    async def health() -> dict:
        asr_service = _get_asr_service()
        model_display = asr_service.funasr_model_path if asr_service.backend == "funasr_nano" else asr_service.model_path
        return {
            "ok": True,
            "asr_backend": asr_service.backend,
            "asr_model": model_display,
            "asr_default_domain": asr_service.default_domain,
            "asr_correction_enabled": asr_service.correction_enabled,
            "tts_enabled": TTS_SERVICE.enabled,
            "tts_vendor_dir": str(TTS_SERVICE.vendor_dir),
            "tts_cache_dir": str(TTS_SERVICE.cache_dir),
        }

    @app.get("/api/asr/backend-options")
    async def asr_backend_options() -> dict:
        asr_service = _get_asr_service()
        return {
            "ok": True,
            "current": _current_backend_selection(asr_service),
            "options": [
                {"value": "whisper", "label": "whisper"},
                {"value": "whisper_small", "label": "whisper_small"},
                {"value": "funasr_nano", "label": "funasr_nano"},
            ],
        }

    @app.post("/api/asr/backend")
    async def set_asr_backend(payload: dict = Body(...)) -> dict:
        selected_backend = str(payload.get("backend", "") or "").strip().lower()
        if selected_backend not in ASR_BACKEND_PRESETS:
            raise HTTPException(status_code=400, detail=f"不支持的 ASR 后端：{selected_backend}")
        try:
            updated_config = _apply_asr_backend_config(selected_backend)
            asr_service = _reload_asr_service(updated_config)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Switch ASR backend failed")
            raise HTTPException(status_code=500, detail=f"切换 ASR 后端失败：{exc}") from exc

        model_display = asr_service.funasr_model_path if asr_service.backend == "funasr_nano" else asr_service.model_path
        return {
            "ok": True,
            "backend": _current_backend_selection(asr_service),
            "runtime_backend": asr_service.backend,
            "model_path": model_display,
        }

    @app.post("/api/asr/transcribe")
    async def transcribe(
        file: Annotated[UploadFile, File(...)],
        language: Annotated[str | None, Form()] = None,
    ) -> dict:
        saved_path = _save_upload(file)
        asr_service = _get_asr_service()
        try:
            result = asr_service.transcribe_file(
                saved_path,
                language=(language or "").strip().lower() or None,
                domain=asr_service.default_domain,
            )
            return {"ok": True, **result, "file_name": saved_path.name}
        except Exception as exc:  # noqa: BLE001
            logger.exception("ASR failed")
            raise HTTPException(status_code=500, detail=f"ASR 失败：{exc}") from exc

    @app.post("/api/asr/transcribe-stream")
    async def transcribe_stream(
        file: Annotated[UploadFile, File(...)],
        language: Annotated[str | None, Form()] = None,
    ) -> StreamingResponse:
        saved_path = _save_upload(file)
        event_queue: queue.Queue[dict] = queue.Queue()
        asr_service = _get_asr_service()

        def push_event(payload: dict) -> None:
            event_queue.put(payload)

        def worker() -> None:
            try:
                result = asr_service.transcribe_file(
                    saved_path,
                    language=(language or "").strip().lower() or None,
                    domain=asr_service.default_domain,
                    progress_callback=push_event,
                )
                event_queue.put({"event": "complete", "ok": True, **result, "file_name": saved_path.name})
            except Exception as exc:  # noqa: BLE001
                logger.exception("ASR stream failed")
                event_queue.put({"event": "error", "ok": False, "detail": f"ASR 失败：{exc}"})
            finally:
                event_queue.put({"event": "close"})

        threading.Thread(target=worker, daemon=True).start()

        def generate():
            while True:
                item = event_queue.get()
                if item.get("event") == "close":
                    break
                yield json.dumps(item, ensure_ascii=False) + "\n"

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @app.post("/api/tts/synthesize")
    async def synthesize(
        text: Annotated[str, Form(...)],
        language: Annotated[str | None, Form()] = None,
        speaker: Annotated[str | None, Form()] = None,
        speed: Annotated[float | None, Form()] = None,
    ) -> dict:
        try:
            result = TTS_SERVICE.synthesize(text=text, language=language, speaker=speaker, speed=speed)
            return {"ok": True, **result}
        except MeloTtsError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("TTS failed")
            raise HTTPException(status_code=500, detail=f"TTS 失败：{exc}") from exc

    @app.get("/media/{file_name}")
    async def media(file_name: str):
        file_path = resolve_path(f"res/tts_output/{file_name}")
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="音频文件不存在。")
        return FileResponse(file_path, media_type="audio/wav", filename=file_name)

    return app


def main() -> int:
    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0
