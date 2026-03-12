from __future__ import annotations

import os
import re
import sys
import threading
import uuid
from typing import Any

from app.config import project_root, resolve_path


class MeloTtsError(RuntimeError):
    pass


class MeloTtsService:
    LANGUAGE_MAP = {
        "auto": "AUTO",
        "zh": "ZH",
        "zh-cn": "ZH",
        "cn": "ZH",
        "en": "EN",
        "en-us": "EN",
        "en-gb": "EN",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        tts_cfg = config.get("tts", {}) if isinstance(config.get("tts"), dict) else {}
        self.enabled = bool(tts_cfg.get("enabled", True))
        self.vendor_dir = resolve_path(str(tts_cfg.get("vendor_dir", "vendor/MeloTTS") or "vendor/MeloTTS"))
        self.cache_dir = resolve_path(str(tts_cfg.get("cache_dir", "models/melotts/cache") or "models/melotts/cache"))
        self.output_dir = resolve_path(str(tts_cfg.get("output_dir", "res/tts_output") or "res/tts_output"))
        self.default_speed = float(tts_cfg.get("speed", 1.0) or 1.0)
        self.language_defaults = tts_cfg.get("default_speakers", {}) if isinstance(tts_cfg.get("default_speakers"), dict) else {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._models: dict[str, Any] = {}

    def _prepare_runtime(self) -> None:
        if self.vendor_dir.is_dir() and str(self.vendor_dir) not in sys.path:
            sys.path.insert(0, str(self.vendor_dir))

        os.environ.setdefault("HF_HOME", str(self.cache_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.cache_dir / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.cache_dir / "transformers"))
        os.environ.setdefault("TORCH_HOME", str(project_root() / "models" / "torch"))

    def _get_tts_class(self):  # noqa: ANN201
        self._prepare_runtime()
        try:
            from melo.api import TTS  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise MeloTtsError(
                "未找到 MeloTTS 运行环境。请将 MeloTTS 源码放入 vendor/MeloTTS，"
                "并在项目内虚拟环境安装其依赖。"
            ) from exc
        return TTS

    def _resolve_language(self, text: str, language: str | None) -> str:
        key = self.LANGUAGE_MAP.get(str(language or "auto").strip().lower(), "AUTO")
        if key != "AUTO":
            return key
        return "ZH" if re.search(r"[\u4e00-\u9fff]", text or "") else "EN"

    def _get_model(self, language: str):  # noqa: ANN201
        if not self.enabled:
            raise MeloTtsError("TTS 已禁用。")

        with self._lock:
            if language in self._models:
                return self._models[language]

            TTS = self._get_tts_class()
            try:
                model = TTS(language=language, device="auto")
            except Exception:
                try:
                    model = TTS(language=language)
                except Exception as exc:  # noqa: BLE001
                    raise MeloTtsError(f"MeloTTS 初始化失败：{exc}") from exc

            self._models[language] = model
            return model

    def _pick_speaker(self, language: str, model, speaker: str | None):  # noqa: ANN001, ANN201
        speaker_ids = getattr(model, "speaker_ids", None)
        if isinstance(speaker_ids, dict) and speaker:
            if speaker in speaker_ids:
                return speaker, speaker_ids[speaker]
            raise MeloTtsError(f"未找到说话人：{speaker}")

        if isinstance(speaker_ids, dict) and speaker_ids:
            configured = str(self.language_defaults.get(language.lower(), "") or "").strip()
            if configured and configured in speaker_ids:
                return configured, speaker_ids[configured]
            first_name = next(iter(speaker_ids))
            return first_name, speaker_ids[first_name]

        data = getattr(getattr(model, "hps", None), "data", None)
        spk2id = getattr(data, "spk2id", None)
        if isinstance(spk2id, dict) and spk2id:
            if speaker and speaker in spk2id:
                return speaker, spk2id[speaker]
            first_name = next(iter(spk2id))
            return first_name, spk2id[first_name]

        return "", 0

    def synthesize(self, *, text: str, language: str | None, speaker: str | None = None, speed: float | None = None) -> dict[str, Any]:
        content = str(text or "").strip()
        if not content:
            raise MeloTtsError("文本不能为空。")

        target_language = self._resolve_language(content, language)
        model = self._get_model(target_language)
        speaker_name, speaker_id = self._pick_speaker(target_language, model, speaker)

        file_name = f"{uuid.uuid4().hex}_{target_language.lower()}.wav"
        output_path = self.output_dir / file_name
        speed_value = float(speed or self.default_speed or 1.0)

        try:
            model.tts_to_file(content, speaker_id, str(output_path), speed=speed_value)
        except Exception as exc:  # noqa: BLE001
            raise MeloTtsError(f"MeloTTS 合成失败：{exc}") from exc

        return {
            "text": content,
            "language": target_language.lower(),
            "speaker": speaker_name,
            "speed": speed_value,
            "file_name": file_name,
            "file_path": str(output_path),
            "audio_url": f"/media/{file_name}",
        }
