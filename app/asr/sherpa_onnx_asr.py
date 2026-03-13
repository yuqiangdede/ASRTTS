from __future__ import annotations

import gc
import hashlib
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SherpaOnnxAsr:
    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cpu",
        hotwords: list[str] | None = None,
        hotwords_score: float = 1.5,
        num_threads: int = 2,
        decoding_method: str = "modified_beam_search",
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.device = (device or "cpu").strip().lower()
        self.default_hotwords = [str(item).strip() for item in (hotwords or []) if str(item).strip()]
        self.hotwords_score = float(hotwords_score or 1.5)
        self.num_threads = max(1, int(num_threads or 1))
        self.decoding_method = str(decoding_method or "modified_beam_search").strip() or "modified_beam_search"
        self._module = None
        self._recognizer = None
        self._recognizer_signature: tuple[str, str] | None = None
        self._active_provider = "cpu"
        self._file_bundle: dict[str, Path] | None = None
        self._lock = threading.Lock()
        self._load_runtime()

    def _load_runtime(self) -> None:
        try:
            import sherpa_onnx  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("导入 sherpa_onnx 失败。请先安装：pip install sherpa-onnx") from exc

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise RuntimeError(
                f"Sherpa ONNX 模型目录不存在：{self.model_path}。"
                "请先执行 scripts\\download_sherpa_onnx_zh_en.py。"
            )

        self._module = sherpa_onnx
        self._file_bundle = self._resolve_model_files(model_dir)
        self._ensure_recognizer(self.default_hotwords)

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        if not path.exists() or not path.is_file():
            return False
        try:
            return path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _first_existing(paths: list[Path]) -> Path | None:
        for item in paths:
            if item.exists() and item.is_file():
                return item
        return None

    @staticmethod
    def _find_model_file(model_dir: Path, stem: str, *, prefer_int8: bool) -> Path | None:
        candidates = [
            item
            for item in model_dir.glob(f"{stem}-*.onnx")
            if item.is_file()
        ]
        if not candidates:
            return None

        def sort_key(path: Path) -> tuple[int, int, str]:
            name = path.name.lower()
            is_int8 = ".int8." in name
            return (
                0 if (prefer_int8 and is_int8) or ((not prefer_int8) and (not is_int8)) else 1,
                0 if is_int8 else 1,
                name,
            )

        candidates.sort(key=sort_key)
        return candidates[0]

    def _resolve_model_files(self, model_dir: Path) -> dict[str, Path]:
        tokens = model_dir / "tokens.txt"
        if not self._is_non_empty_file(tokens):
            raise RuntimeError(
                f"Sherpa ONNX 缺少 tokens.txt：{tokens}。"
                "请重新执行 scripts\\download_sherpa_onnx_zh_en.py。"
            )

        prefer_int8 = self.device != "cuda"
        encoder = self._find_model_file(model_dir, "encoder", prefer_int8=prefer_int8)
        decoder = self._find_model_file(model_dir, "decoder", prefer_int8=prefer_int8)
        joiner = self._find_model_file(model_dir, "joiner", prefer_int8=prefer_int8)
        missing: list[str] = []
        if encoder is None:
            missing.append("encoder-*.onnx")
        if decoder is None:
            missing.append("decoder-*.onnx")
        if joiner is None:
            missing.append("joiner-*.onnx")
        if missing:
            raise RuntimeError(
                "Sherpa ONNX 模型目录缺少必需文件："
                + ", ".join(missing)
                + f"。当前目录：{model_dir}"
            )
        return {
            "tokens": tokens,
            "encoder": encoder,
            "decoder": decoder,
            "joiner": joiner,
            "bbpe_model": model_dir / "bbpe.model",
        }

    @staticmethod
    def _normalize_hotwords(default_hotwords: list[str], hotwords: list[str] | None) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for item in [*default_hotwords, *(hotwords or [])]:
            value = str(item).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
        return merged

    @staticmethod
    def _fallback_encode_hotword(text: str, *, use_bpe: bool) -> str:
        compact = "".join(str(text).split())
        if not compact:
            return ""
        if not use_bpe:
            return " ".join(list(compact))

        pattern = re.compile(r"([\u4e00-\u9fff])")
        pieces = pattern.split(compact)
        tokens: list[str] = []
        for piece in pieces:
            value = piece.strip()
            if not value:
                continue
            if pattern.fullmatch(value) is not None:
                tokens.append(value)
            else:
                tokens.extend(part for part in value.split() if part)
                if " " not in value:
                    tokens.append(value)
        return " ".join(dict.fromkeys(token for token in tokens if token))

    def _encode_hotwords_lines(self, hotwords: list[str]) -> list[str]:
        if self._module is None or self._file_bundle is None:
            raise RuntimeError("Sherpa ONNX 运行时尚未初始化。")

        bbpe_model = self._file_bundle.get("bbpe_model")
        use_bpe = bbpe_model is not None and bbpe_model.is_file()
        encoder = getattr(self._module, "text2token", None)
        if callable(encoder):
            try:
                token_lists = encoder(
                    hotwords,
                    tokens=str(self._file_bundle["tokens"]),
                    tokens_type="cjkchar+bpe" if use_bpe else "cjkchar",
                    bpe_model=str(bbpe_model) if use_bpe else None,
                    output_ids=False,
                )
                encoded_lines = [" ".join(str(token).strip() for token in tokens if str(token).strip()) for tokens in token_lists]
                encoded_lines = [line for line in encoded_lines if line]
                if encoded_lines:
                    return encoded_lines
            except Exception as exc:  # noqa: BLE001
                logger.warning("Sherpa ONNX 热词编码失败，将回退到简化编码：error=%s", exc)

        fallback_lines = [
            self._fallback_encode_hotword(item, use_bpe=use_bpe)
            for item in hotwords
        ]
        return [line for line in fallback_lines if line]

    def _build_hotwords_file(self, hotwords: list[str]) -> Path | None:
        if not hotwords:
            return None
        lines = self._encode_hotwords_lines(hotwords)
        if not lines:
            return None
        hotword_dir = Path(self.model_path) / ".hotwords"
        hotword_dir.mkdir(parents=True, exist_ok=True)
        signature = hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        target = hotword_dir / f"hotwords-{signature}.txt"
        if not target.exists():
            target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def _provider_candidates(self) -> list[str]:
        if self.device == "cuda":
            return ["cuda", "cpu"]
        return ["cpu"]

    def _build_recognizer(self, *, provider: str, hotwords: list[str]) -> Any:
        if self._module is None or self._file_bundle is None:
            raise RuntimeError("Sherpa ONNX 运行时尚未初始化。")

        module = self._module
        recognizer_cls = getattr(module, "OfflineRecognizer", None)
        factory = getattr(recognizer_cls, "from_transducer", None) if recognizer_cls is not None else None
        if not callable(factory):
            raise RuntimeError("当前 sherpa_onnx 版本缺少离线识别所需的 Python API。")

        hotwords_file = self._build_hotwords_file(hotwords)
        recognizer_kwargs: dict[str, Any] = {
            "encoder": str(self._file_bundle["encoder"]),
            "decoder": str(self._file_bundle["decoder"]),
            "joiner": str(self._file_bundle["joiner"]),
            "tokens": str(self._file_bundle["tokens"]),
            "num_threads": self.num_threads,
            "sample_rate": 16000,
            "feature_dim": 80,
            "dither": 0.0,
            "decoding_method": self.decoding_method,
            "max_active_paths": 4,
            "blank_penalty": 0.0,
            "debug": False,
            "provider": provider,
            "model_type": "transducer",
        }
        if hotwords_file is not None:
            recognizer_kwargs["hotwords_file"] = str(hotwords_file)
            recognizer_kwargs["hotwords_score"] = self.hotwords_score
        return factory(**recognizer_kwargs)

    def _ensure_recognizer(self, hotwords: list[str]) -> None:
        signature = ("|".join(hotwords), self.device)
        with self._lock:
            if self._recognizer is not None and self._recognizer_signature == signature:
                return

            last_error: Exception | None = None
            for provider in self._provider_candidates():
                try:
                    recognizer = self._build_recognizer(provider=provider, hotwords=hotwords)
                    self._recognizer = recognizer
                    self._recognizer_signature = signature
                    self._active_provider = provider
                    logger.info(
                        "Sherpa ONNX 模型已加载：model=%s provider=%s encoder=%s",
                        self.model_path,
                        provider,
                        self._file_bundle["encoder"].name if self._file_bundle else "",
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    logger.warning("Sherpa ONNX 初始化失败，将尝试其他 provider：provider=%s error=%s", provider, exc)

            raise RuntimeError(f"Sherpa ONNX 模型加载失败：{last_error}") from last_error

    @staticmethod
    def _extract_text(result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result.strip()
        value = getattr(result, "text", None)
        if isinstance(value, str):
            return value.strip()
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                return text.strip()
        return str(result).strip()

    @staticmethod
    def _stream_result_text(stream: Any, recognizer: Any | None = None) -> str:
        stream_result = getattr(stream, "result", None)
        if stream_result is not None:
            return SherpaOnnxAsr._extract_text(stream_result)
        getter = getattr(stream, "get_result", None)
        if callable(getter):
            return SherpaOnnxAsr._extract_text(getter())
        recognizer_getter = getattr(recognizer, "get_result", None)
        if callable(recognizer_getter):
            return SherpaOnnxAsr._extract_text(recognizer_getter(stream))
        inner_recognizer = getattr(recognizer, "recognizer", None)
        inner_getter = getattr(inner_recognizer, "get_result", None)
        if callable(inner_getter):
            return SherpaOnnxAsr._extract_text(inner_getter(stream))
        return ""

    @staticmethod
    def _accept_waveform(stream: Any, *, sample_rate: int, waveform: np.ndarray) -> None:
        try:
            stream.accept_waveform(sample_rate, waveform)
        except TypeError:
            stream.accept_waveform(sample_rate=sample_rate, waveform=waveform)

    def transcribe_pcm16(
        self,
        pcm16_bytes: bytes,
        *,
        sample_rate: int = 16000,
        hotwords: list[str] | None = None,
        source_label: str = "pcm16",
    ) -> dict[str, Any]:
        merged_hotwords = self._normalize_hotwords(self.default_hotwords, hotwords)
        self._ensure_recognizer(merged_hotwords)
        if self._recognizer is None:
            raise RuntimeError("Sherpa ONNX 模型未初始化。")

        waveform = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
        if waveform.size == 0:
            raise RuntimeError("Sherpa ONNX 输入音频为空。")
        waveform /= 32768.0

        logger.info(
            "Sherpa ONNX 推理开始：source=%s samples=%s hotwords=%s provider=%s",
            source_label,
            waveform.size,
            len(merged_hotwords),
            self._active_provider,
        )
        stream = self._recognizer.create_stream()
        self._accept_waveform(stream, sample_rate=sample_rate, waveform=waveform)
        input_finished = getattr(stream, "input_finished", None)
        if callable(input_finished):
            input_finished()
        decode_stream = getattr(self._recognizer, "decode_stream", None)
        if callable(decode_stream):
            decode_stream(stream)
        else:
            decode = getattr(self._recognizer, "decode", None)
            if not callable(decode):
                raise RuntimeError("当前 sherpa_onnx 版本缺少 decode_stream/decode 接口。")
            decode(stream)
        text = self._stream_result_text(stream, self._recognizer)
        logger.info(
            "Sherpa ONNX transcribe finished: source=%s hotwords=%s text=%s",
            source_label,
            ",".join(merged_hotwords[:50]),
            text[:200],
        )
        return {
            "text": text,
            "language": "zh-en",
            "language_probability": 1.0,
            "hotwords": merged_hotwords,
        }

    def close(self) -> None:
        self._recognizer = None
        self._recognizer_signature = None
        gc.collect()
