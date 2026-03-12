from __future__ import annotations

import gc
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FunAsrNano:
    def __init__(
        self,
        *,
        model_path: str,
        vad_model_path: str | None = None,
        device: str = "cuda",
        hotwords: list[str] | None = None,
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.vad_model_path = str(Path(vad_model_path).resolve()) if vad_model_path else ""
        self.device = (device or "cuda").strip().lower()
        self.default_hotwords = [str(item).strip() for item in (hotwords or []) if str(item).strip()]
        self._model = None
        self._model_no_vad = None
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

    def _runtime_dir(self) -> Path:
        return Path(self.model_path) / "runtime"

    def _remote_code_path(self) -> Path:
        return self._runtime_dir() / "model.py"

    def _required_runtime_files(self) -> list[Path]:
        runtime_dir = self._runtime_dir()
        return [
            runtime_dir / "model.py",
            runtime_dir / "ctc.py",
            runtime_dir / "decode.py",
            runtime_dir / "tools" / "utils.py",
        ]

    def _load_model(self) -> None:
        try:
            from funasr import AutoModel  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("导入 funasr 失败。请先安装：pip install funasr") from exc
        self._auto_model_class = AutoModel

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(
                f"FunASR 模型目录不存在：{self.model_path}。"
                "请先执行 scripts\\download_funasr_nano.py。"
            )
        missing_runtime_files = [path for path in self._required_runtime_files() if not path.is_file()]
        if missing_runtime_files:
            missing_str = "，".join(str(path) for path in missing_runtime_files)
            raise RuntimeError(
                f"FunASR 运行时代码缺失：{missing_str}。"
                "请重新执行 scripts\\download_funasr_nano.py。"
            )

        runtime_dir = str(self._runtime_dir().resolve())
        if runtime_dir not in sys.path:
            sys.path.insert(0, runtime_dir)

        try:
            self._model = self._create_model(device=self.device, use_vad=True)
            logger.info(
                "FunASR Nano 模型已加载：model=%s requested_device=%s actual_device=%s",
                self.model_path,
                self.device,
                self._actual_model_device(self._model),
            )
        except UnboundLocalError as exc:
            if "get_tokenizer" in str(exc):
                raise RuntimeError(
                    "FunASR 依赖的 tokenizer 未安装完整。请先执行："
                    ".venv\\Scripts\\python.exe -m pip install openai-whisper"
                ) from exc
            raise
        except Exception as exc:  # noqa: BLE001
            if self.device != "cpu":
                logger.warning("FunASR Nano GPU 初始化失败，将回退 CPU：%s", exc)
                self.device = "cpu"
                try:
                    self._model = self._create_model(device="cpu", use_vad=True)
                    logger.info(
                        "FunASR Nano 模型已加载：model=%s requested_device=cpu actual_device=%s",
                        self.model_path,
                        self._actual_model_device(self._model),
                    )
                    return
                except UnboundLocalError as inner_exc:
                    if "get_tokenizer" in str(inner_exc):
                        raise RuntimeError(
                            "FunASR 依赖的 tokenizer 未安装完整。请先执行："
                            ".venv\\Scripts\\python.exe -m pip install openai-whisper"
                        ) from inner_exc
                    raise
            raise RuntimeError(f"FunASR 模型加载失败：{exc}") from exc

    def _create_model(self, *, device: str, use_vad: bool) -> Any:
        if self._auto_model_class is None:
            raise RuntimeError("FunASR AutoModel 未初始化。")
        remote_code = self._remote_code_path().resolve().as_posix()
        kwargs: dict[str, Any] = {
            "model": self.model_path,
            "trust_remote_code": True,
            "remote_code": remote_code,
            "disable_update": True,
            "device": device,
        }
        if use_vad:
            kwargs["vad_model"] = self._ensure_local_vad_model()
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
        return self._auto_model_class(**kwargs)

    def _ensure_local_vad_model(self) -> str:
        if not self.vad_model_path:
            raise RuntimeError("未配置 FunASR VAD 模型路径。")

        target = Path(self.vad_model_path).resolve()
        if target.is_dir():
            return str(target)

        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        if cache_dir.is_dir():
            logger.warning("检测到 VAD 模型在用户缓存中，将复制到项目目录：%s -> %s", cache_dir, target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cache_dir, target, dirs_exist_ok=True)
            return str(target)

        raise RuntimeError(
            f"FunASR VAD 模型不存在：{target}。请执行 scripts\\download_funasr_vad.py，"
            "或执行 scripts\\download_funasr_nano.py 下载安装整套 FunASR 资源。"
        )

    def _ensure_model_no_vad(self) -> Any:
        if self._model_no_vad is None:
            logger.warning("FunASR Nano VAD 推理失败，将回退无 VAD 模式重试。")
            self._model_no_vad = self._create_model(device=self.device, use_vad=False)
        return self._model_no_vad

    @staticmethod
    def _merge_hotwords(default_hotwords: list[str], hotwords: list[str] | None) -> list[str]:
        merged_hotwords = [*default_hotwords]
        for item in hotwords or []:
            value = str(item).strip()
            if value and value not in merged_hotwords:
                merged_hotwords.append(value)
        return merged_hotwords

    def transcribe_file(self, audio_path: str | Path, *, hotwords: list[str] | None = None) -> dict[str, Any]:
        return self.transcribe_file_with_options(audio_path, hotwords=hotwords, prefer_no_vad=False)

    def transcribe_pcm16(
        self,
        pcm16_bytes: bytes,
        *,
        sample_rate: int = 16000,
        hotwords: list[str] | None = None,
        prefer_no_vad: bool = False,
        source_label: str = "pcm16",
    ) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("FunASR 模型未初始化。")

        waveform_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if waveform_np.size == 0:
            raise RuntimeError("FunASR 输入音频为空。")
        waveform_np /= 32768.0
        waveform = torch.from_numpy(waveform_np)

        merged_hotwords = self._merge_hotwords(self.default_hotwords, hotwords)
        kwargs: dict[str, Any] = {
            "input": [waveform],
            "cache": {},
            "batch_size": 1,
            "language": "中文",
            "itn": True,
        }
        model = self._ensure_model_no_vad() if prefer_no_vad else self._model
        try:
            result = self._generate(model=model, kwargs=kwargs, hotwords=merged_hotwords)
        except Exception as exc:  # noqa: BLE001
            if prefer_no_vad:
                raise
            if self._should_retry_without_vad(exc):
                result = self._generate(model=self._ensure_model_no_vad(), kwargs=kwargs, hotwords=merged_hotwords)
            else:
                raise

        text = self._extract_text(result)
        logger.info(
            "FunASR Nano transcribe finished: source=%s hotwords=%s text=%s",
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

    def transcribe_file_with_options(
        self,
        audio_path: str | Path,
        *,
        hotwords: list[str] | None = None,
        prefer_no_vad: bool = False,
    ) -> dict[str, Any]:
        from app.audio import decode_audio_to_pcm16

        audio_file = str(Path(audio_path).resolve())
        pcm16_bytes, _, _ = decode_audio_to_pcm16(audio_file, sample_rate=16000)
        return self.transcribe_pcm16(
            pcm16_bytes,
            sample_rate=16000,
            hotwords=hotwords,
            prefer_no_vad=prefer_no_vad,
            source_label=audio_file,
        )

    def _generate(self, *, model: Any, kwargs: dict[str, Any], hotwords: list[str]) -> Any:
        if model is None:
            raise RuntimeError("FunASR 模型未初始化。")

        if not hotwords:
            return model.generate(**kwargs)

        attempts = [
            {**kwargs, "hotwords": hotwords},
            {**kwargs, "hotword": " ".join(hotwords)},
        ]
        last_error: Exception | None = None
        for payload in attempts:
            try:
                return model.generate(**payload)
            except TypeError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise RuntimeError(f"FunASR 热词参数不兼容：{last_error}") from last_error
        return model.generate(**kwargs)

    @staticmethod
    def _should_retry_without_vad(exc: Exception) -> bool:
        message = str(exc or "")
        return isinstance(exc, KeyError) or "KeyError: 0" in message or "timestamp" in message.lower()

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
        self._model_no_vad = None
        self._auto_model_class = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass



