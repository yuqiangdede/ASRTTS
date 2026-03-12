from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ParaformerAsr:
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        hotwords: list[str] | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.device = (device or "cuda").strip().lower()
        self.default_hotwords = [str(item).strip() for item in (hotwords or []) if str(item).strip()]
        self._model = None
        self._auto_model_class = None
        self._load_model()

    @staticmethod
    def _actual_model_device(model: Any) -> str:
        try:
            inner = getattr(model, "model", None)
            if inner is None:
                return "unknown"
            param = next(inner.parameters())
            return str(param.device)
        except Exception:
            return "unknown"

    def _load_model(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("导入 funasr 失败。请先安装：pip install funasr") from exc

        self._auto_model_class = AutoModel
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Paraformer 模型目录不存在：{self.model_path}。"
                "请先执行 scripts\\download_paraformer.py。"
            )

        try:
            self._model = self._create_model(device=self.device)
            logger.info(
                "Paraformer 模型已加载：model=%s requested_device=%s actual_device=%s",
                self.model_path,
                self.device,
                self._actual_model_device(self._model),
            )
        except Exception as exc:  # noqa: BLE001
            if self.device != "cpu":
                logger.warning("Paraformer GPU 初始化失败，将回退 CPU：%s", exc)
                self.device = "cpu"
                self._model = self._create_model(device="cpu")
                logger.info(
                    "Paraformer 模型已加载：model=%s requested_device=cpu actual_device=%s",
                    self.model_path,
                    self._actual_model_device(self._model),
                )
                return
            raise RuntimeError(f"Paraformer 模型加载失败：{exc}") from exc

    def _create_model(self, *, device: str) -> Any:
        if self._auto_model_class is None:
            raise RuntimeError("Paraformer AutoModel 未初始化。")
        return self._auto_model_class(
            model=self.model_path,
            disable_update=True,
            device=device,
        )

    @staticmethod
    def _merge_hotwords(default_hotwords: list[str], hotwords: list[str] | None) -> list[str]:
        merged_hotwords = [*default_hotwords]
        for item in hotwords or []:
            value = str(item).strip()
            if value and value not in merged_hotwords:
                merged_hotwords.append(value)
        return merged_hotwords

    def transcribe_pcm16(
        self,
        pcm16_bytes: bytes,
        *,
        sample_rate: int = 16000,
        hotwords: list[str] | None = None,
        source_label: str = "pcm16",
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("Paraformer 模型未初始化。")

        waveform = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if waveform.size == 0:
            raise RuntimeError("Paraformer 输入音频为空。")
        waveform /= 32768.0

        merged_hotwords = self._merge_hotwords(self.default_hotwords, hotwords)
        payload: dict[str, Any] = {
            "input": waveform,
            "fs": sample_rate,
            "cache": {},
            "batch_size_s": 300,
            "use_itn": True,
        }
        result = self._generate(payload=payload, hotwords=merged_hotwords)
        text = self._extract_text(result)
        logger.info(
            "Paraformer transcribe finished: source=%s hotwords=%s text=%s",
            source_label,
            ",".join(merged_hotwords[:50]),
            text[:200],
        )
        return {
            "text": text,
            "language": "zh",
            "language_probability": 1.0,
            "hotwords": merged_hotwords,
        }

    def _generate(self, *, payload: dict[str, Any], hotwords: list[str]) -> Any:
        if self._model is None:
            raise RuntimeError("Paraformer 模型未初始化。")

        attempts: list[dict[str, Any]] = [{**payload}]
        if hotwords:
            attempts = [
                {**payload, "hotword": " ".join(hotwords)},
                {**payload, "hotwords": hotwords},
                {**payload},
            ]

        last_error: Exception | None = None
        for index, current_payload in enumerate(attempts, start=1):
            try:
                return self._model.generate(**current_payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if index < len(attempts):
                    logger.warning("Paraformer 当前热词参数尝试失败，将自动切换参数形式：%s", exc)
                    continue
                raise RuntimeError(f"Paraformer 推理失败：{exc}") from exc

        if last_error is not None:
            raise RuntimeError(f"Paraformer 推理失败：{last_error}") from last_error
        raise RuntimeError("Paraformer 推理失败：未知错误。")

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            value = result.get("text")
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                return "".join(str(item) for item in value).strip()
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                value = first.get("text")
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, list):
                    return "".join(str(item) for item in value).strip()
            return "".join(str(item) for item in result).strip()
        return str(result or "").strip()

    def close(self) -> None:
        self._model = None
        self._auto_model_class = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
