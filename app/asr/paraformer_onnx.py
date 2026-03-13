from __future__ import annotations

import gc
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)


class ParaformerOnnxAsr:
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
        self._paraformer_class = None
        self._contextual_mode = False
        self._compat_dir = Path(self.model_path) / ".runtime_compat"
        self._load_model()

    @staticmethod
    def _detect_runtime_device(requested_device: str) -> str:
        providers = set(ort.get_available_providers())
        if requested_device == "cuda" and "CUDAExecutionProvider" in providers:
            return "cuda"
        return "cpu"

    def _ensure_compat_layout(self) -> Path:
        source_dir = Path(self.model_path)
        compat_dir = self._compat_dir
        compat_dir.mkdir(parents=True, exist_ok=True)

        passthrough_files = [
            "am.mvn",
            "config.yaml",
            "configuration.json",
            "tokens.json",
            "tokens.txt",
            ".mv",
            ".msc",
            "asr.yaml",
            "asr.json",
            "gen_tokens.py",
        ]
        for relative_name in passthrough_files:
            source_file = source_dir / relative_name
            target_file = compat_dir / relative_name
            if source_file.is_file() and (not target_file.exists() or target_file.stat().st_size == 0):
                shutil.copy2(source_file, target_file)

        model_aliases = {
            "model.int8.onnx": ["model_quant.onnx", "model.onnx"],
            "model_eb.int8.onnx": ["model_eb_quant.onnx", "model_eb.onnx"],
        }
        for source_name, target_names in model_aliases.items():
            source_file = source_dir / source_name
            if not source_file.is_file():
                continue
            for target_name in target_names:
                target_file = compat_dir / target_name
                if target_file.exists() and target_file.stat().st_size > 0:
                    continue
                shutil.copy2(source_file, target_file)

        source_hotword = source_dir / "hotword.txt"
        hotword_file = compat_dir / "hotword.txt"
        if source_hotword.is_file() and (not hotword_file.exists() or hotword_file.stat().st_size == 0):
            shutil.copy2(source_hotword, hotword_file)

        return compat_dir

    def _load_model(self) -> None:
        try:
            from funasr_onnx import Paraformer, SeacoParaformer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "导入 funasr_onnx 失败。请先安装：pip install funasr-onnx。"
            ) from exc

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Paraformer ONNX 模型目录不存在：{self.model_path}。"
                "请先执行 scripts\\download_paraformer_onnx.py。"
            )

        compat_dir = self._ensure_compat_layout()
        actual_device = self._detect_runtime_device(self.device)
        self.device = actual_device
        device_id = 0 if actual_device == "cuda" else -1
        source_dir = Path(self.model_path)
        is_seaco = "seaco" in source_dir.name.lower() or (source_dir / "model_eb.int8.onnx").is_file()
        self._paraformer_class = SeacoParaformer if is_seaco else Paraformer
        self._contextual_mode = is_seaco

        try:
            self._model = self._paraformer_class(
                str(compat_dir),
                batch_size=1,
                quantize=True,
                device_id=device_id,
            )
        except TypeError:
            self._model = self._paraformer_class(
                str(compat_dir),
                batch_size=1,
                quantize=True,
            )

        logger.info(
            "Paraformer ONNX 模型已加载：model=%s runtime_dir=%s actual_device=%s contextual=%s",
            self.model_path,
            compat_dir,
            actual_device,
            self._contextual_mode,
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
        merged_hotwords = self._merge_hotwords(self.default_hotwords, hotwords)
        if self._model is None:
            raise RuntimeError("Paraformer ONNX 模型未初始化。")

        waveform = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if waveform.size == 0:
            raise RuntimeError("Paraformer ONNX 输入音频为空。")
        waveform = waveform / 32768.0
        logger.info(
            "Paraformer ONNX 推理开始：source=%s samples=%s hotwords=%s contextual=%s device=%s",
            source_label,
            waveform.size,
            len(merged_hotwords),
            self._contextual_mode,
            self.device,
        )

        text = self._run_inference(waveform, merged_hotwords)
        if not text:
            logger.warning(
                "Paraformer ONNX 首次结果为空，尝试使用临时 wav 回退：source=%s",
                source_label,
            )
            text = self._run_inference_via_temp_wav(pcm16_bytes, sample_rate, merged_hotwords)

        logger.info(
            "Paraformer ONNX transcribe finished: source=%s hotwords=%s text=%s",
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

    def _run_inference(self, waveform: np.ndarray, hotwords: list[str]) -> str:
        if self._model is None:
            raise RuntimeError("Paraformer ONNX 模型未初始化。")
        if self._contextual_mode:
            hotword_text = " ".join(item for item in hotwords if item.strip())
            result = self._model(waveform, hotwords=hotword_text)
        else:
            result = self._model(waveform)
        return self._extract_text(result)

    def _run_inference_via_temp_wav(self, pcm16_bytes: bytes, sample_rate: int, hotwords: list[str]) -> str:
        if self._model is None:
            raise RuntimeError("Paraformer ONNX 模型未初始化。")
        temp_path = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        try:
            import wave

            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm16_bytes)
            if self._contextual_mode:
                hotword_text = " ".join(item for item in hotwords if item.strip())
                result = self._model(str(temp_path), hotwords=hotword_text)
            else:
                result = self._model(str(temp_path))
            return self._extract_text(result)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                logger.warning("删除 Paraformer ONNX 临时 wav 失败：%s", temp_path, exc_info=True)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, list):
            if not result:
                return ""
            first = result[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                value = first.get("preds") or first.get("text")
                if isinstance(value, str):
                    return value.strip()
                if isinstance(value, list):
                    return "".join(str(item) for item in value).strip()
        if isinstance(result, dict):
            value = result.get("preds") or result.get("text")
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                return "".join(str(item) for item in value).strip()
        return str(result or "").strip()

    def close(self) -> None:
        self._model = None
        self._paraformer_class = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
