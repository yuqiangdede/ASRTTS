from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.audio import decode_audio_to_pcm16, split_pcm16_by_silence
from app.asr.correction import AsrCorrectionClient
from app.asr.funasr_nano import FunAsrNano
from app.asr.paraformer import ParaformerAsr
from app.asr.paraformer_onnx import ParaformerOnnxAsr
from app.asr.sherpa_onnx_asr import SherpaOnnxAsr
from app.asr.whisper import WhisperAsr
from app.config import resolve_path
from config_loader import DomainConfigLoader
from pipeline import AsrEnhancementPipeline
from phrase_corrector import PhraseCorrector

logger = logging.getLogger(__name__)


class AsrService:
    def __init__(self, config: dict[str, Any]) -> None:
        asr_cfg = config.get("asr", {}) if isinstance(config.get("asr"), dict) else {}
        post_cfg = config.get("postprocess", {}) if isinstance(config.get("postprocess"), dict) else {}

        self.backend = str(asr_cfg.get("backend", "whisper") or "whisper").strip().lower()
        model_path = str(asr_cfg.get("model_path", "faster-whisper-small") or "faster-whisper-small").strip()
        device = str(asr_cfg.get("device", "cuda") or "cuda").strip().lower()
        compute_type = str(asr_cfg.get("compute_type", "int8_float32") or "int8_float32").strip().lower()
        cuda_bin_dir = str(asr_cfg.get("cuda_bin_dir", "") or "").strip() or None
        suspicious = asr_cfg.get("cuda_suspicious_min_chars_per_s", 1)
        funasr_cfg = asr_cfg.get("funasr_nano", {}) if isinstance(asr_cfg.get("funasr_nano"), dict) else {}

        model_resolved = resolve_path(model_path)
        self.model_path = str(model_resolved if model_resolved.exists() else Path(model_path))
        self.device = device
        self.compute_type = compute_type
        self.cuda_bin_dir = cuda_bin_dir
        self.suspicious = float(suspicious) if suspicious is not None else None
        self.auto_fallback_language = str(asr_cfg.get("auto_fallback_language", "zh") or "zh").strip().lower()
        self.auto_fallback_prob_threshold = float(asr_cfg.get("auto_fallback_prob_threshold", 0.70) or 0.70)
        self.auto_fallback_allowed_langs = {
            str(item).strip().lower()
            for item in asr_cfg.get("auto_fallback_allowed_langs", ["zh", "en"])
            if str(item).strip()
        }
        self.funasr_model_path = str(resolve_path(str(funasr_cfg.get("model_path", "models/Fun-ASR-Nano-2512") or "models/Fun-ASR-Nano-2512")))
        self.funasr_vad_model_path = str(resolve_path(str(funasr_cfg.get("vad_model_path", "models/speech_fsmn_vad_zh-cn-16k-common-pytorch") or "models/speech_fsmn_vad_zh-cn-16k-common-pytorch")))
        self.funasr_hotwords = [str(item).strip() for item in funasr_cfg.get("hotwords", []) if str(item).strip()]
        self.funasr_use_prompt_terms_as_hotwords = bool(funasr_cfg.get("use_prompt_terms_as_hotwords", True))
        self.funasr_hotword_mode = str(funasr_cfg.get("hotword_mode", "compact") or "compact").strip().lower()
        self.funasr_max_hotwords = int(funasr_cfg.get("max_hotwords", 1000) or 1000)
        self.funasr_release_after_inference = bool(funasr_cfg.get("release_after_inference", False))
        paraformer_cfg = asr_cfg.get("paraformer", {}) if isinstance(asr_cfg.get("paraformer"), dict) else {}
        self.paraformer_model_path = str(
            resolve_path(
                str(
                    paraformer_cfg.get(
                        "model_path",
                        "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    )
                    or "models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                )
            )
        )
        self.paraformer_hotwords = [str(item).strip() for item in paraformer_cfg.get("hotwords", []) if str(item).strip()]
        self.paraformer_use_prompt_terms_as_hotwords = bool(paraformer_cfg.get("use_prompt_terms_as_hotwords", True))
        self.paraformer_hotword_mode = str(paraformer_cfg.get("hotword_mode", "full") or "full").strip().lower()
        self.paraformer_max_hotwords = int(paraformer_cfg.get("max_hotwords", 1000) or 1000)
        paraformer_onnx_cfg = asr_cfg.get("paraformer_onnx", {}) if isinstance(asr_cfg.get("paraformer_onnx"), dict) else {}
        self.paraformer_onnx_model_path = str(
            resolve_path(
                str(
                    paraformer_onnx_cfg.get(
                        "model_path",
                        "models/paraformer-seaco-large-zh-timestamp-int8-onnx-offline",
                    )
                    or "models/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
                )
            )
        )
        self.paraformer_onnx_hotwords = [
            str(item).strip() for item in paraformer_onnx_cfg.get("hotwords", []) if str(item).strip()
        ]
        self.paraformer_onnx_use_prompt_terms_as_hotwords = bool(
            paraformer_onnx_cfg.get("use_prompt_terms_as_hotwords", True)
        )
        self.paraformer_onnx_hotword_mode = str(
            paraformer_onnx_cfg.get("hotword_mode", "full") or "full"
        ).strip().lower()
        self.paraformer_onnx_max_hotwords = int(paraformer_onnx_cfg.get("max_hotwords", 1000) or 1000)
        sherpa_onnx_cfg = asr_cfg.get("sherpa_onnx", {}) if isinstance(asr_cfg.get("sherpa_onnx"), dict) else {}
        self.sherpa_onnx_model_path = str(
            resolve_path(
                str(
                    sherpa_onnx_cfg.get(
                        "model_path",
                        "models/sherpa-onnx-zipformer-zh-en-2023-11-22",
                    )
                    or "models/sherpa-onnx-zipformer-zh-en-2023-11-22"
                )
            )
        )
        self.sherpa_onnx_hotwords = [
            str(item).strip() for item in sherpa_onnx_cfg.get("hotwords", []) if str(item).strip()
        ]
        self.sherpa_onnx_use_prompt_terms_as_hotwords = bool(
            sherpa_onnx_cfg.get("use_prompt_terms_as_hotwords", True)
        )
        self.sherpa_onnx_hotword_mode = str(
            sherpa_onnx_cfg.get("hotword_mode", "compact") or "compact"
        ).strip().lower()
        self.sherpa_onnx_max_hotwords = int(sherpa_onnx_cfg.get("max_hotwords", 1000) or 1000)
        self.sherpa_onnx_hotwords_score = float(sherpa_onnx_cfg.get("hotwords_score", 1.5) or 1.5)
        self.sherpa_onnx_num_threads = int(sherpa_onnx_cfg.get("num_threads", 2) or 2)
        self.sherpa_onnx_decoding_method = str(
            sherpa_onnx_cfg.get("decoding_method", "modified_beam_search") or "modified_beam_search"
        ).strip()

        self.domain_profiles_path = resolve_path(str(post_cfg.get("domain_profiles_path", "domain_profiles.yaml") or "domain_profiles.yaml"))
        self.common_terms_path = resolve_path(str(post_cfg.get("common_terms_path", "security_terms.yaml") or "security_terms.yaml"))
        self.default_domain = str(post_cfg.get("default_domain", "prison") or "prison").strip()
        correction_cfg = config.get("asr_correction", {}) if isinstance(config.get("asr_correction"), dict) else {}
        self.correction_client = AsrCorrectionClient(correction_cfg)
        self.correction_enabled = self.correction_client.enabled

        self.pipeline = self._build_pipeline()
        self.text_normalizer = PhraseCorrector()

        self._lock = threading.Lock()
        self._postprocess_signature = self._current_postprocess_signature()
        self._model: WhisperAsr | None = None
        self._funasr_model: FunAsrNano | None = None
        self._paraformer_model: ParaformerAsr | None = None
        self._paraformer_onnx_model: ParaformerOnnxAsr | None = None
        self._sherpa_onnx_model: SherpaOnnxAsr | None = None

    def _build_correction_mode_hint(self, *, hotwords: list[str] | None = None) -> str:
        asr_mode = "识别阶段：普通识别"
        hotword_count = len(hotwords or [])
        model_name = Path(self.model_path).name.lower()
        if self.backend == "whisper":
            if "small" in model_name:
                asr_mode = "识别阶段：Whisper Small，会参考业务词"
            else:
                asr_mode = "识别阶段：Whisper Large，会参考业务词"
        elif self.backend == "funasr_nano":
            asr_mode = f"识别阶段：FunASR Nano，会重点参考 {hotword_count} 个热词"
        elif self.backend == "paraformer":
            asr_mode = f"识别阶段：Paraformer，会重点参考 {hotword_count} 个热词"
        elif self.backend == "paraformer_onnx":
            asr_mode = f"识别阶段：Paraformer ONNX，会重点参考 {hotword_count} 个热词"
        elif self.backend == "sherpa_onnx":
            asr_mode = f"识别阶段：Sherpa ONNX 中英双语，会重点参考 {hotword_count} 个热词"

        return f"{asr_mode}；识别后会继续做：去掉异常空格、按业务词纠正、再结合上下文细化"

    def _whisper_download_script(self) -> str:
        model_name = Path(self.model_path).name.lower()
        if "small" in model_name:
            return "scripts\\download_whisper_small.py"
        return "scripts\\download_whisper_large_v3_turbo.py"

    def list_domains(self) -> list[str]:
        self._refresh_postprocess_pipeline_if_needed()
        return self.pipeline.list_domains()

    def _build_pipeline(self) -> AsrEnhancementPipeline:
        return AsrEnhancementPipeline(
            DomainConfigLoader(self.domain_profiles_path, self.common_terms_path),
            default_domain=self.default_domain,
            correction_client=self.correction_client,
        )

    @staticmethod
    def _safe_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            return None

    def _current_postprocess_signature(self) -> tuple[float | None, float | None]:
        return (
            self._safe_mtime(self.domain_profiles_path),
            self._safe_mtime(self.common_terms_path),
        )

    def _refresh_postprocess_pipeline_if_needed(self) -> None:
        signature = self._current_postprocess_signature()
        with self._lock:
            if signature == self._postprocess_signature:
                return
            self.pipeline = self._build_pipeline()
            self._postprocess_signature = signature
        logger.info(
            "检测到后处理配置文件变更，已自动重载：domain_profiles=%s security_terms=%s",
            self.domain_profiles_path,
            self.common_terms_path,
        )

    def _get_model(self) -> WhisperAsr:
        with self._lock:
            if self._model is None:
                if not Path(self.model_path).exists():
                    raise RuntimeError(
                        f"Whisper 模型不存在：{self.model_path}。"
                        f"请先执行 {self._whisper_download_script()}。"
                    )
                self._model = WhisperAsr(
                    model=self.model_path,
                    device=self.device,
                    cuda_bin_dir=self.cuda_bin_dir,
                    compute_type=self.compute_type,
                    cuda_suspicious_min_chars_per_s=self.suspicious,
                )
            return self._model

    def _get_funasr_model(self) -> FunAsrNano:
        with self._lock:
            if self._funasr_model is None:
                if not Path(self.funasr_model_path).exists():
                    raise RuntimeError(
                        f"FunASR 模型不存在：{self.funasr_model_path}。"
                        "请先执行 scripts\\download_funasr_nano.py。"
                    )
                if not Path(self.funasr_vad_model_path).exists():
                    raise RuntimeError(
                        f"FunASR VAD 模型不存在：{self.funasr_vad_model_path}。"
                        "请先执行 scripts\\download_funasr_nano.py。"
                    )
                self._funasr_model = FunAsrNano(
                    model_path=self.funasr_model_path,
                    vad_model_path=self.funasr_vad_model_path,
                    device=self.device,
                    hotwords=self.funasr_hotwords,
                )
            return self._funasr_model

    def _get_paraformer_model(self) -> ParaformerAsr:
        with self._lock:
            if self._paraformer_model is None:
                if not Path(self.paraformer_model_path).exists():
                    raise RuntimeError(
                        f"Paraformer 模型不存在：{self.paraformer_model_path}。"
                        "请先执行 scripts\\download_paraformer.py。"
                    )
                self._paraformer_model = ParaformerAsr(
                    model_path=self.paraformer_model_path,
                    device=self.device,
                    hotwords=self.paraformer_hotwords,
                )
            return self._paraformer_model

    def _get_paraformer_onnx_model(self) -> ParaformerOnnxAsr:
        with self._lock:
            if self._paraformer_onnx_model is None:
                if not Path(self.paraformer_onnx_model_path).exists():
                    raise RuntimeError(
                        f"Paraformer ONNX 模型不存在：{self.paraformer_onnx_model_path}。"
                        "请先执行 scripts\\download_paraformer_onnx.py。"
                    )
                self._paraformer_onnx_model = ParaformerOnnxAsr(
                    model_path=self.paraformer_onnx_model_path,
                    device=self.device,
                    hotwords=self.paraformer_onnx_hotwords,
                )
            return self._paraformer_onnx_model

    def _get_sherpa_onnx_model(self) -> SherpaOnnxAsr:
        with self._lock:
            if self._sherpa_onnx_model is None:
                if not Path(self.sherpa_onnx_model_path).exists():
                    raise RuntimeError(
                        f"Sherpa ONNX 模型不存在：{self.sherpa_onnx_model_path}。"
                        "请先执行 scripts\\download_sherpa_onnx_zh_en.py。"
                    )
                self._sherpa_onnx_model = SherpaOnnxAsr(
                    model_path=self.sherpa_onnx_model_path,
                    device=self.device,
                    hotwords=self.sherpa_onnx_hotwords,
                    hotwords_score=self.sherpa_onnx_hotwords_score,
                    num_threads=self.sherpa_onnx_num_threads,
                    decoding_method=self.sherpa_onnx_decoding_method,
                )
            return self._sherpa_onnx_model

    def _release_whisper_model(self) -> None:
        with self._lock:
            model = self._model
            self._model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                logger.warning("释放 Whisper 模型失败", exc_info=True)

    def _release_funasr_model(self) -> None:
        with self._lock:
            model = self._funasr_model
            self._funasr_model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                logger.warning("释放 FunASR 模型失败", exc_info=True)

    def _release_paraformer_model(self) -> None:
        with self._lock:
            model = self._paraformer_model
            self._paraformer_model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                logger.warning("释放 Paraformer 模型失败", exc_info=True)

    def _release_paraformer_onnx_model(self) -> None:
        with self._lock:
            model = self._paraformer_onnx_model
            self._paraformer_onnx_model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                logger.warning("释放 Paraformer ONNX 模型失败", exc_info=True)

    def _release_sherpa_onnx_model(self) -> None:
        with self._lock:
            model = self._sherpa_onnx_model
            self._sherpa_onnx_model = None
        if model is not None and hasattr(model, "close"):
            try:
                model.close()
            except Exception:
                logger.warning("释放 Sherpa ONNX 模型失败", exc_info=True)

    def close(self) -> None:
        self._release_whisper_model()
        self._release_funasr_model()
        self._release_paraformer_model()
        self._release_paraformer_onnx_model()
        self._release_sherpa_onnx_model()

    def _should_retry_with_fallback(self, *, explicit_language: str | None, result_text: str, result_language: str, result_prob: float) -> bool:
        if explicit_language:
            return False
        if not self.auto_fallback_language:
            return False
        if not result_text:
            return True
        if result_language and result_language in self.auto_fallback_allowed_langs and result_prob >= self.auto_fallback_prob_threshold:
            return False
        return result_prob < self.auto_fallback_prob_threshold

    def _transcribe_once(
        self,
        *,
        pcm16_bytes: bytes,
        sample_rate: int,
        language: str | None,
        initial_prompt: str | None,
    ):
        return self._get_model().transcribe(
            pcm16_bytes=pcm16_bytes,
            sample_rate=sample_rate,
            language=language,
            initial_prompt=initial_prompt,
        )

    def transcribe_file(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        domain: str | None = None,
        enable_vad: bool = False,
        enable_split: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        self._refresh_postprocess_pipeline_if_needed()
        active_domain = domain or self.default_domain
        initial_prompt = self.pipeline.build_initial_prompt(active_domain)
        profile = self.pipeline.config_loader.get_profile(active_domain)

        if self.backend == "funasr_nano":
            return self._transcribe_file_with_funasr(
                audio_path=audio_path,
                active_domain=active_domain,
                initial_prompt=initial_prompt,
                profile_prompt_terms=profile.prompt_terms,
                enable_vad=enable_vad,
                enable_split=enable_split,
                progress_callback=progress_callback,
            )
        if self.backend == "paraformer":
            return self._transcribe_file_with_paraformer(
                audio_path=audio_path,
                active_domain=active_domain,
                initial_prompt=initial_prompt,
                profile_prompt_terms=profile.prompt_terms,
                enable_vad=enable_vad,
                enable_split=enable_split,
                progress_callback=progress_callback,
            )
        if self.backend == "paraformer_onnx":
            return self._transcribe_file_with_paraformer_onnx(
                audio_path=audio_path,
                active_domain=active_domain,
                initial_prompt=initial_prompt,
                profile_prompt_terms=profile.prompt_terms,
                enable_vad=enable_vad,
                enable_split=enable_split,
                progress_callback=progress_callback,
            )
        if self.backend == "sherpa_onnx":
            return self._transcribe_file_with_sherpa_onnx(
                audio_path=audio_path,
                active_domain=active_domain,
                initial_prompt=initial_prompt,
                profile_prompt_terms=profile.prompt_terms,
                enable_vad=enable_vad,
                enable_split=enable_split,
                progress_callback=progress_callback,
            )

        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        explicit_language = (language or "").strip().lower() or None
        asr_started = time.perf_counter()
        if enable_split:
            segment_pcm_list = self._split_pcm_for_request(
                backend="whisper",
                source_path=audio_path,
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            result = self._transcribe_whisper_segments(
                segment_pcm_list=segment_pcm_list,
                sample_rate=sample_rate,
                language=explicit_language,
                initial_prompt=(initial_prompt or None),
            )
        else:
            result = self._transcribe_once(
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                language=explicit_language,
                initial_prompt=(initial_prompt or None),
            )
        used_fallback_language = ""
        raw_text_original = result.text.strip()
        raw_text = self.text_normalizer.normalize_text(raw_text_original)
        result_language = (result.language or "").strip().lower()
        result_prob = float(result.language_probability or 0.0)

        if self._should_retry_with_fallback(
            explicit_language=explicit_language,
            result_text=raw_text,
            result_language=result_language,
            result_prob=result_prob,
        ):
            retry_result = self._transcribe_once(
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                language=self.auto_fallback_language,
                initial_prompt=(initial_prompt or None),
            )
            retry_text = retry_result.text.strip()
            if retry_text:
                result = retry_result
                raw_text_original = retry_text
                raw_text = self.text_normalizer.normalize_text(raw_text_original)
                result_language = (retry_result.language or "").strip().lower() or self.auto_fallback_language
                result_prob = float(retry_result.language_probability or 0.0)
                used_fallback_language = self.auto_fallback_language
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        logger.info("ASR raw original text: backend=%s text=%s", self.backend, raw_text_original[:500])

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
                "raw_text_original": raw_text_original,
                "raw_text": raw_text,
                "backend": self.backend,
                "domain": active_domain,
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": used_fallback_language,
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "hotwords": [],
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=[]),
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)

        return {
            "backend": self.backend,
            "text": processed["final_text"],
            "raw_text_original": processed["raw_text_original"],
            "raw_text": processed["raw_text"],
            "text_after_phrase": processed["text_after_phrase"],
            "final_text": processed["final_text"],
            "applied_rules": processed["applied_rules"],
            "domain": processed["domain"],
            "llm_correction_applied": processed["llm_correction_applied"],
            "correction_error": processed["correction_error"],
            "initial_prompt": initial_prompt,
            "language": result_language,
            "language_probability": result_prob,
            "used_fallback_language": used_fallback_language,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "asr_elapsed_ms": asr_elapsed_ms,
            "phrase_elapsed_ms": processed["phrase_elapsed_ms"],
            "confusion_elapsed_ms": processed["confusion_elapsed_ms"],
            "llm_elapsed_ms": processed["llm_elapsed_ms"],
            "postprocess_elapsed_ms": processed["postprocess_elapsed_ms"],
            "hotwords": [],
            "enable_vad": enable_vad,
            "enable_split": enable_split,
            "correction_mode_hint": self._build_correction_mode_hint(hotwords=[]),
        }

    def _transcribe_file_with_funasr(
        self,
        *,
        audio_path: str | Path,
        active_domain: str,
        initial_prompt: str,
        profile_prompt_terms: list[str],
        enable_vad: bool,
        enable_split: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        hotwords = [*self.funasr_hotwords]
        if self.funasr_use_prompt_terms_as_hotwords:
            selected_terms = self._select_funasr_hotwords(profile_prompt_terms)
            for term in selected_terms:
                if term not in hotwords:
                    hotwords.append(term)

        asr_started = time.perf_counter()
        if enable_split:
            segment_pcm_list = self._split_pcm_for_request(
                backend="funasr_nano",
                source_path=audio_path,
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            result = self._transcribe_funasr_segments(
                source_path=audio_path,
                segment_pcm_list=segment_pcm_list,
                sample_rate=sample_rate,
                hotwords=hotwords,
                prefer_no_vad=not enable_vad,
            )
        else:
            result = self._get_funasr_model().transcribe_pcm16(
                pcm16_bytes,
                sample_rate=sample_rate,
                hotwords=hotwords,
                prefer_no_vad=not enable_vad,
                source_label=str(Path(audio_path).resolve()),
            )
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        raw_text_original = str(result.get("text", "") or "").strip()
        raw_text = self.text_normalizer.normalize_text(raw_text_original)
        result_language = str(result.get("language", "zh") or "zh").strip().lower()
        result_prob = float(result.get("language_probability", 1.0) or 1.0)
        logger.info("ASR raw original text: backend=%s text=%s", self.backend, raw_text_original[:500])

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
                "raw_text_original": raw_text_original,
                "raw_text": raw_text,
                "backend": self.backend,
                "domain": active_domain,
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": "",
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "hotwords": hotwords,
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
            },
        )

        try:
            processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)
            return {
                "backend": self.backend,
                "text": processed["final_text"],
                "raw_text_original": processed["raw_text_original"],
                "raw_text": processed["raw_text"],
                "text_after_phrase": processed["text_after_phrase"],
                "final_text": processed["final_text"],
                "applied_rules": processed["applied_rules"],
                "domain": processed["domain"],
                "llm_correction_applied": processed["llm_correction_applied"],
                "correction_error": processed["correction_error"],
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": "",
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "phrase_elapsed_ms": processed["phrase_elapsed_ms"],
                "confusion_elapsed_ms": processed["confusion_elapsed_ms"],
                "llm_elapsed_ms": processed["llm_elapsed_ms"],
                "postprocess_elapsed_ms": processed["postprocess_elapsed_ms"],
                "hotwords": hotwords,
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
            }
        finally:
            if self.funasr_release_after_inference:
                logger.info("FunASR 推理完成，按配置释放模型与显存。")
                self._release_funasr_model()

    def _transcribe_file_with_paraformer(
        self,
        *,
        audio_path: str | Path,
        active_domain: str,
        initial_prompt: str,
        profile_prompt_terms: list[str],
        enable_vad: bool,
        enable_split: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        hotwords = [*self.paraformer_hotwords]
        if self.paraformer_use_prompt_terms_as_hotwords:
            selected_terms = self._select_hotwords(
                profile_prompt_terms,
                mode=self.paraformer_hotword_mode,
                max_count=self.paraformer_max_hotwords,
            )
            for term in selected_terms:
                if term not in hotwords:
                    hotwords.append(term)

        asr_started = time.perf_counter()
        if enable_split:
            segment_pcm_list = self._split_pcm_for_request(
                backend="paraformer",
                source_path=audio_path,
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            result = self._transcribe_paraformer_segments(
                model=self._get_paraformer_model(),
                source_path=audio_path,
                segment_pcm_list=segment_pcm_list,
                sample_rate=sample_rate,
                hotwords=hotwords,
            )
        else:
            result = self._get_paraformer_model().transcribe_pcm16(
                pcm16_bytes,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(Path(audio_path).resolve()),
            )
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        raw_text_original = str(result.get("text", "") or "").strip()
        raw_text = self.text_normalizer.normalize_text(raw_text_original)
        result_language = str(result.get("language", "zh") or "zh").strip().lower()
        result_prob = float(result.get("language_probability", 1.0) or 1.0)
        logger.info("ASR raw original text: backend=%s text=%s", self.backend, raw_text_original[:500])

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
                "raw_text_original": raw_text_original,
                "raw_text": raw_text,
                "backend": self.backend,
                "domain": active_domain,
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": "",
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "hotwords": hotwords,
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)
        return {
            "backend": self.backend,
            "text": processed["final_text"],
            "raw_text_original": processed["raw_text_original"],
            "raw_text": processed["raw_text"],
            "text_after_phrase": processed["text_after_phrase"],
            "final_text": processed["final_text"],
            "applied_rules": processed["applied_rules"],
            "domain": processed["domain"],
            "llm_correction_applied": processed["llm_correction_applied"],
            "correction_error": processed["correction_error"],
            "initial_prompt": initial_prompt,
            "language": result_language,
            "language_probability": result_prob,
            "used_fallback_language": "",
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "asr_elapsed_ms": asr_elapsed_ms,
            "phrase_elapsed_ms": processed["phrase_elapsed_ms"],
            "confusion_elapsed_ms": processed["confusion_elapsed_ms"],
            "llm_elapsed_ms": processed["llm_elapsed_ms"],
            "postprocess_elapsed_ms": processed["postprocess_elapsed_ms"],
            "hotwords": hotwords,
            "enable_vad": enable_vad,
            "enable_split": enable_split,
            "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
        }

    def _transcribe_file_with_paraformer_onnx(
        self,
        *,
        audio_path: str | Path,
        active_domain: str,
        initial_prompt: str,
        profile_prompt_terms: list[str],
        enable_vad: bool,
        enable_split: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        hotwords = [*self.paraformer_onnx_hotwords]
        if self.paraformer_onnx_use_prompt_terms_as_hotwords:
            selected_terms = self._select_hotwords(
                profile_prompt_terms,
                mode=self.paraformer_onnx_hotword_mode,
                max_count=self.paraformer_onnx_max_hotwords,
            )
            for term in selected_terms:
                if term not in hotwords:
                    hotwords.append(term)

        asr_started = time.perf_counter()
        if enable_split:
            segment_pcm_list = self._split_pcm_for_request(
                backend="paraformer_onnx",
                source_path=audio_path,
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            result = self._transcribe_paraformer_onnx_segments(
                model=self._get_paraformer_onnx_model(),
                source_path=audio_path,
                segment_pcm_list=segment_pcm_list,
                sample_rate=sample_rate,
                hotwords=hotwords,
            )
        else:
            result = self._get_paraformer_onnx_model().transcribe_pcm16(
                pcm16_bytes,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(Path(audio_path).resolve()),
            )
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        raw_text_original = str(result.get("text", "") or "").strip()
        raw_text = self.text_normalizer.normalize_text(raw_text_original)
        result_language = str(result.get("language", "zh") or "zh").strip().lower()
        result_prob = float(result.get("language_probability", 1.0) or 1.0)
        logger.info("ASR raw original text: backend=%s text=%s", self.backend, raw_text_original[:500])

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
                "raw_text_original": raw_text_original,
                "raw_text": raw_text,
                "backend": self.backend,
                "domain": active_domain,
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": "",
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "hotwords": hotwords,
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)
        return {
            "backend": self.backend,
            "text": processed["final_text"],
            "raw_text_original": processed["raw_text_original"],
            "raw_text": processed["raw_text"],
            "text_after_phrase": processed["text_after_phrase"],
            "final_text": processed["final_text"],
            "applied_rules": processed["applied_rules"],
            "domain": processed["domain"],
            "llm_correction_applied": processed["llm_correction_applied"],
            "correction_error": processed["correction_error"],
            "initial_prompt": initial_prompt,
            "language": result_language,
            "language_probability": result_prob,
            "used_fallback_language": "",
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "asr_elapsed_ms": asr_elapsed_ms,
            "phrase_elapsed_ms": processed["phrase_elapsed_ms"],
            "confusion_elapsed_ms": processed["confusion_elapsed_ms"],
            "llm_elapsed_ms": processed["llm_elapsed_ms"],
            "postprocess_elapsed_ms": processed["postprocess_elapsed_ms"],
            "hotwords": hotwords,
            "enable_vad": enable_vad,
            "enable_split": enable_split,
            "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
        }

    def _transcribe_file_with_sherpa_onnx(
        self,
        *,
        audio_path: str | Path,
        active_domain: str,
        initial_prompt: str,
        profile_prompt_terms: list[str],
        enable_vad: bool,
        enable_split: bool,
        progress_callback: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        hotwords = [*self.sherpa_onnx_hotwords]
        if self.sherpa_onnx_use_prompt_terms_as_hotwords:
            selected_terms = self._select_hotwords(
                profile_prompt_terms,
                mode=self.sherpa_onnx_hotword_mode,
                max_count=self.sherpa_onnx_max_hotwords,
            )
            for term in selected_terms:
                if term not in hotwords:
                    hotwords.append(term)

        asr_started = time.perf_counter()
        if enable_split:
            segment_pcm_list = self._split_pcm_for_request(
                backend="sherpa_onnx",
                source_path=audio_path,
                pcm16_bytes=pcm16_bytes,
                sample_rate=sample_rate,
                duration_s=duration_s,
            )
            result = self._transcribe_sherpa_onnx_segments(
                model=self._get_sherpa_onnx_model(),
                source_path=audio_path,
                segment_pcm_list=segment_pcm_list,
                sample_rate=sample_rate,
                hotwords=hotwords,
            )
        else:
            result = self._get_sherpa_onnx_model().transcribe_pcm16(
                pcm16_bytes,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(Path(audio_path).resolve()),
            )
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        raw_text_original = str(result.get("text", "") or "").strip()
        raw_text = self.text_normalizer.normalize_text(raw_text_original)
        result_language = str(result.get("language", "zh-en") or "zh-en").strip().lower()
        result_prob = float(result.get("language_probability", 1.0) or 1.0)
        logger.info("ASR raw original text: backend=%s text=%s", self.backend, raw_text_original[:500])

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
                "raw_text_original": raw_text_original,
                "raw_text": raw_text,
                "backend": self.backend,
                "domain": active_domain,
                "initial_prompt": initial_prompt,
                "language": result_language,
                "language_probability": result_prob,
                "used_fallback_language": "",
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "asr_elapsed_ms": asr_elapsed_ms,
                "hotwords": hotwords,
                "enable_vad": enable_vad,
                "enable_split": enable_split,
                "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)
        return {
            "backend": self.backend,
            "text": processed["final_text"],
            "raw_text_original": processed["raw_text_original"],
            "raw_text": processed["raw_text"],
            "text_after_phrase": processed["text_after_phrase"],
            "final_text": processed["final_text"],
            "applied_rules": processed["applied_rules"],
            "domain": processed["domain"],
            "llm_correction_applied": processed["llm_correction_applied"],
            "correction_error": processed["correction_error"],
            "initial_prompt": initial_prompt,
            "language": result_language,
            "language_probability": result_prob,
            "used_fallback_language": "",
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "asr_elapsed_ms": asr_elapsed_ms,
            "phrase_elapsed_ms": processed["phrase_elapsed_ms"],
            "confusion_elapsed_ms": processed["confusion_elapsed_ms"],
            "llm_elapsed_ms": processed["llm_elapsed_ms"],
            "postprocess_elapsed_ms": processed["postprocess_elapsed_ms"],
            "hotwords": hotwords,
            "enable_vad": enable_vad,
            "enable_split": enable_split,
            "correction_mode_hint": self._build_correction_mode_hint(hotwords=hotwords),
        }

    def _transcribe_funasr_segments(
        self,
        *,
        source_path: str | Path,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        hotwords: list[str],
        prefer_no_vad: bool,
    ) -> dict[str, Any]:
        model = self._get_funasr_model()
        source = Path(source_path).resolve()
        if len(segment_pcm_list) < 2:
            return model.transcribe_pcm16(
                segment_pcm_list[0],
                sample_rate=sample_rate,
                hotwords=hotwords,
                prefer_no_vad=prefer_no_vad,
                source_label=str(source),
            )

        segment_texts: list[str] = []
        for index, segment_pcm in enumerate(segment_pcm_list, start=1):
            segment_result = model.transcribe_pcm16(
                segment_pcm,
                sample_rate=sample_rate,
                hotwords=hotwords,
                prefer_no_vad=prefer_no_vad,
                source_label=f"{source}#segment{index}",
            )
            segment_text = str(segment_result.get("text", "") or "").strip()
            if segment_text:
                segment_texts.append(segment_text)
            logger.info(
                "ASR segmented chunk finished: index=%s/%s text=%s",
                index,
                len(segment_pcm_list),
                segment_text[:200],
            )

        final_text = self._merge_segment_texts(segment_texts)
        return {
            "text": final_text,
            "language": "zh",
            "language_probability": 1.0,
            "hotwords": hotwords,
        }

    def _transcribe_paraformer_segments(
        self,
        *,
        model: ParaformerAsr,
        source_path: str | Path,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        source = Path(source_path).resolve()
        if len(segment_pcm_list) < 2:
            return model.transcribe_pcm16(
                segment_pcm_list[0],
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(source),
            )

        segment_texts: list[str] = []
        for index, segment_pcm in enumerate(segment_pcm_list, start=1):
            segment_result = model.transcribe_pcm16(
                segment_pcm,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=f"{source}#segment{index}",
            )
            segment_text = str(segment_result.get("text", "") or "").strip()
            if segment_text:
                segment_texts.append(segment_text)
            logger.info("ASR segmented chunk finished: index=%s/%s text=%s", index, len(segment_pcm_list), segment_text[:200])

        return {
            "text": self._merge_segment_texts(segment_texts),
            "language": "zh",
            "language_probability": 1.0,
            "hotwords": hotwords,
        }

    def _transcribe_paraformer_onnx_segments(
        self,
        *,
        model: ParaformerOnnxAsr,
        source_path: str | Path,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        source = Path(source_path).resolve()
        if len(segment_pcm_list) < 2:
            return model.transcribe_pcm16(
                segment_pcm_list[0],
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(source),
            )

        segment_texts: list[str] = []
        for index, segment_pcm in enumerate(segment_pcm_list, start=1):
            segment_result = model.transcribe_pcm16(
                segment_pcm,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=f"{source}#segment{index}",
            )
            segment_text = str(segment_result.get("text", "") or "").strip()
            if segment_text:
                segment_texts.append(segment_text)
            logger.info("ASR segmented chunk finished: index=%s/%s text=%s", index, len(segment_pcm_list), segment_text[:200])

        return {
            "text": self._merge_segment_texts(segment_texts),
            "language": "zh",
            "language_probability": 1.0,
            "hotwords": hotwords,
        }

    def _transcribe_sherpa_onnx_segments(
        self,
        *,
        model: SherpaOnnxAsr,
        source_path: str | Path,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        source = Path(source_path).resolve()
        if len(segment_pcm_list) < 2:
            return model.transcribe_pcm16(
                segment_pcm_list[0],
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=str(source),
            )

        segment_texts: list[str] = []
        for index, segment_pcm in enumerate(segment_pcm_list, start=1):
            segment_result = model.transcribe_pcm16(
                segment_pcm,
                sample_rate=sample_rate,
                hotwords=hotwords,
                source_label=f"{source}#segment{index}",
            )
            segment_text = str(segment_result.get("text", "") or "").strip()
            if segment_text:
                segment_texts.append(segment_text)
            logger.info("ASR segmented chunk finished: index=%s/%s text=%s", index, len(segment_pcm_list), segment_text[:200])

        return {
            "text": self._merge_segment_texts(segment_texts),
            "language": "zh-en",
            "language_probability": 1.0,
            "hotwords": hotwords,
        }

    def _transcribe_whisper_segments(
        self,
        *,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        language: str | None,
        initial_prompt: str | None,
    ):
        if len(segment_pcm_list) < 2:
            return self._transcribe_once(
                pcm16_bytes=segment_pcm_list[0],
                sample_rate=sample_rate,
                language=language,
                initial_prompt=initial_prompt,
            )

        segment_results = [
            self._transcribe_once(
                pcm16_bytes=segment_pcm,
                sample_rate=sample_rate,
                language=language,
                initial_prompt=initial_prompt,
            )
            for segment_pcm in segment_pcm_list
        ]
        merged_text = self._merge_segment_texts([item.text for item in segment_results])
        first = next((item for item in segment_results if item.text.strip()), segment_results[0])
        return first.__class__(
            text=merged_text,
            language=first.language,
            language_probability=first.language_probability,
        )

    def _split_pcm_for_request(
        self,
        *,
        backend: str,
        source_path: str | Path,
        pcm16_bytes: bytes,
        sample_rate: int,
        duration_s: float,
    ) -> list[bytes]:
        segment_pcm_list = split_pcm16_by_silence(pcm16_bytes, sample_rate=sample_rate)
        logger.info(
            "ASR custom silence split: backend=%s source=%s segments=%s duration=%.2fs",
            backend,
            source_path,
            len(segment_pcm_list),
            duration_s,
        )
        return segment_pcm_list

    @staticmethod
    def _merge_segment_texts(parts: list[str]) -> str:
        cleaned = [str(item).strip().strip("，,。.;； ") for item in parts if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "，".join(cleaned)

    def _select_funasr_hotwords(self, profile_prompt_terms: list[str]) -> list[str]:
        return self._select_hotwords(
            profile_prompt_terms,
            mode=self.funasr_hotword_mode,
            max_count=self.funasr_max_hotwords,
        )

    def _select_hotwords(self, profile_prompt_terms: list[str], *, mode: str, max_count: int) -> list[str]:
        terms = [str(item).strip() for item in profile_prompt_terms if str(item).strip()]
        if mode == "full":
            return terms

        prioritized: list[str] = []
        for term in terms:
            if self._is_priority_hotword(term):
                prioritized.append(term)

        seen: set[str] = set()
        result: list[str] = []
        for term in [*prioritized, *terms]:
            if term in seen:
                continue
            seen.add(term)
            result.append(term)
            if len(result) >= max_count:
                break
        return result

    @staticmethod
    def _is_priority_hotword(term: str) -> bool:
        value = str(term or "").strip()
        if not value:
            return False
        keywords = (
            "监区",
            "监墙",
            "周界",
            "点位",
            "门",
            "门禁",
            "通道",
            "会见楼",
            "监舍",
            "监门",
            "车间",
            "武警",
            "警察",
        )
        return any(keyword in value for keyword in keywords)

    def _emit_progress(self, callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if callback is None:
            return
        callback(payload)




