from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from app.audio import decode_audio_to_pcm16, split_pcm16_by_silence
from app.asr.correction import AsrCorrectionClient
from app.asr.funasr_nano import FunAsrNano
from app.asr.whisper import WhisperAsr
from app.config import resolve_path
from config_loader import DomainConfigLoader
from pipeline import AsrEnhancementPipeline

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
        self.funasr_max_hotwords = int(funasr_cfg.get("max_hotwords", 24) or 24)

        self.domain_profiles_path = resolve_path(str(post_cfg.get("domain_profiles_path", "domain_profiles.yaml") or "domain_profiles.yaml"))
        self.common_terms_path = resolve_path(str(post_cfg.get("common_terms_path", "security_terms.yaml") or "security_terms.yaml"))
        self.default_domain = str(post_cfg.get("default_domain", "prison") or "prison").strip()
        correction_cfg = config.get("asr_correction", {}) if isinstance(config.get("asr_correction"), dict) else {}
        self.correction_client = AsrCorrectionClient(correction_cfg)
        self.correction_enabled = self.correction_client.enabled

        self.pipeline = AsrEnhancementPipeline(
            DomainConfigLoader(self.domain_profiles_path, self.common_terms_path),
            default_domain=self.default_domain,
            correction_client=self.correction_client,
        )

        self._lock = threading.Lock()
        self._model: WhisperAsr | None = None
        self._funasr_model: FunAsrNano | None = None

    def _whisper_download_script(self) -> str:
        model_name = Path(self.model_path).name.lower()
        if "small" in model_name:
            return "scripts\\download_whisper_small.py"
        return "scripts\\download_whisper_large_v3_turbo.py"

    def list_domains(self) -> list[str]:
        return self.pipeline.list_domains()

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
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        active_domain = domain or self.default_domain
        initial_prompt = self.pipeline.build_initial_prompt(active_domain)
        profile = self.pipeline.config_loader.get_profile(active_domain)

        if self.backend == "funasr_nano":
            return self._transcribe_file_with_funasr(
                audio_path=audio_path,
                active_domain=active_domain,
                initial_prompt=initial_prompt,
                profile_prompt_terms=profile.prompt_terms,
                progress_callback=progress_callback,
            )

        pcm16_bytes, sample_rate, duration_s = decode_audio_to_pcm16(audio_path, sample_rate=16000)
        explicit_language = (language or "").strip().lower() or None
        asr_started = time.perf_counter()
        result = self._transcribe_once(
            pcm16_bytes=pcm16_bytes,
            sample_rate=sample_rate,
            language=explicit_language,
            initial_prompt=(initial_prompt or None),
        )
        used_fallback_language = ""
        raw_text = result.text.strip()
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
                raw_text = retry_text
                result_language = (retry_result.language or "").strip().lower() or self.auto_fallback_language
                result_prob = float(retry_result.language_probability or 0.0)
                used_fallback_language = self.auto_fallback_language
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
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
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)

        return {
            "backend": self.backend,
            "text": processed["final_text"],
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
        }

    def _transcribe_file_with_funasr(
        self,
        *,
        audio_path: str | Path,
        active_domain: str,
        initial_prompt: str,
        profile_prompt_terms: list[str],
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
        segment_pcm_list = split_pcm16_by_silence(pcm16_bytes, sample_rate=sample_rate)
        logger.info(
            "ASR custom silence split: backend=funasr_nano source=%s segments=%s duration=%.2fs",
            audio_path,
            len(segment_pcm_list),
            duration_s,
        )
        result = self._transcribe_funasr_segments(
            source_path=audio_path,
            segment_pcm_list=segment_pcm_list,
            sample_rate=sample_rate,
            hotwords=hotwords,
        )
        asr_elapsed_ms = (time.perf_counter() - asr_started) * 1000.0
        raw_text = str(result.get("text", "") or "").strip()
        result_language = str(result.get("language", "zh") or "zh").strip().lower()
        result_prob = float(result.get("language_probability", 1.0) or 1.0)

        self._emit_progress(
            progress_callback,
            {
                "event": "raw_text",
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
            },
        )

        processed = self.pipeline.process_text(raw_text, domain=active_domain, progress_callback=progress_callback)
        return {
            "backend": self.backend,
            "text": processed["final_text"],
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
        }

    def _transcribe_funasr_segments(
        self,
        *,
        source_path: str | Path,
        segment_pcm_list: list[bytes],
        sample_rate: int,
        hotwords: list[str],
    ) -> dict[str, Any]:
        model = self._get_funasr_model()
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
                prefer_no_vad=True,
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

    @staticmethod
    def _merge_segment_texts(parts: list[str]) -> str:
        cleaned = [str(item).strip().strip("，,。.;； ") for item in parts if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        return "，".join(cleaned)

    def _select_funasr_hotwords(self, profile_prompt_terms: list[str]) -> list[str]:
        terms = [str(item).strip() for item in profile_prompt_terms if str(item).strip()]
        if self.funasr_hotword_mode == "full":
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
            if len(result) >= self.funasr_max_hotwords:
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




