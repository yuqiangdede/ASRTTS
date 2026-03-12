from __future__ import annotations

import logging
import time
from typing import Any, Callable

from config_loader import DomainConfigLoader
from domain_corrector import DomainCorrector
from phrase_corrector import PhraseCorrector


logger = logging.getLogger(__name__)


class AsrEnhancementPipeline:
    def __init__(self, config_loader: DomainConfigLoader, *, default_domain: str, correction_client: Any | None = None) -> None:
        self.config_loader = config_loader
        self.default_domain = default_domain
        self.phrase_corrector = PhraseCorrector()
        self.domain_corrector = DomainCorrector()
        self.correction_client = correction_client

    def list_domains(self) -> list[str]:
        return self.config_loader.list_domains()

    def build_initial_prompt(self, domain: str | None = None) -> str:
        target_domain = domain or self.default_domain
        return self.config_loader.build_initial_prompt(target_domain)

    def process_text(
        self,
        raw_text: str,
        *,
        domain: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict:
        target_domain = domain or self.default_domain
        profile = self.config_loader.get_profile(target_domain)

        phrase_started = time.perf_counter()
        text_after_phrase, phrase_rules = self.phrase_corrector.apply(raw_text, profile.phrase_rules)
        phrase_elapsed_ms = (time.perf_counter() - phrase_started) * 1000.0
        self._emit_progress(
            progress_callback,
            {
                "event": "phrase_text",
                "domain": profile.name,
                "text_after_phrase": text_after_phrase,
                "applied_rules": phrase_rules,
                "phrase_elapsed_ms": phrase_elapsed_ms,
            },
        )
        confusion_started = time.perf_counter()
        text_after_confusion, confusion_rules = self.domain_corrector.apply(text_after_phrase, profile.confusion_rules)
        confusion_elapsed_ms = (time.perf_counter() - confusion_started) * 1000.0
        llm_started = time.perf_counter()
        final_text, llm_rules, correction_error = self._apply_llm_correction(text_after_confusion, profile)
        llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
        postprocess_elapsed_ms = phrase_elapsed_ms + confusion_elapsed_ms + llm_elapsed_ms
        self._emit_progress(
            progress_callback,
            {
                "event": "final_text",
                "domain": profile.name,
                "final_text": final_text,
                "applied_rules": [*phrase_rules, *confusion_rules, *llm_rules],
                "llm_correction_applied": bool(llm_rules),
                "correction_error": correction_error,
                "confusion_elapsed_ms": confusion_elapsed_ms,
                "llm_elapsed_ms": llm_elapsed_ms,
                "postprocess_elapsed_ms": postprocess_elapsed_ms,
            },
        )

        return {
            "raw_text": raw_text,
            "text_after_phrase": text_after_phrase,
            "final_text": final_text,
            "applied_rules": [*phrase_rules, *confusion_rules, *llm_rules],
            "domain": profile.name,
            "llm_correction_applied": bool(llm_rules),
            "correction_error": correction_error,
            "phrase_elapsed_ms": phrase_elapsed_ms,
            "confusion_elapsed_ms": confusion_elapsed_ms,
            "llm_elapsed_ms": llm_elapsed_ms,
            "postprocess_elapsed_ms": postprocess_elapsed_ms,
        }

    def _emit_progress(self, callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
        if callback is None:
            return
        callback(payload)

    def _apply_llm_correction(self, text: str, profile) -> tuple[str, list[dict[str, Any]], str]:
        current = str(text or "").strip()
        if not current or self.correction_client is None or not getattr(self.correction_client, "enabled", False):
            return current, [], ""

        try:
            corrected = str(
                self.correction_client.correct(
                    current,
                    domain=profile.name,
                    prompt_terms=profile.prompt_terms,
                )
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASR LLM correction failed: domain=%s error=%s", profile.name, exc)
            return current, [], str(exc)

        if not corrected or corrected == current:
            logger.info("ASR LLM correction no change: domain=%s text=%s", profile.name, current[:200])
            return current, [], ""

        protected_terms = self._collect_protected_terms(current, getattr(profile, "prompt_terms", []))
        lost_terms = [term for term in protected_terms if term not in corrected]
        if lost_terms:
            message = f"大模型结果触发业务词保护，已回退：{', '.join(lost_terms[:10])}"
            logger.warning(
                "ASR LLM correction rejected by protected terms: domain=%s protected=%s before=%s after=%s",
                profile.name,
                ",".join(lost_terms[:20]),
                current[:200],
                corrected[:200],
            )
            return current, [], message

        logger.info(
            "ASR LLM correction applied: domain=%s before=%s after=%s",
            profile.name,
            current[:200],
            corrected[:200],
        )

        return (
            corrected,
            [
                {
                    "stage": "llm",
                    "rule": "asr_llm_correction",
                    "before": current,
                    "after": corrected,
                    "replacement": corrected,
                }
            ],
            "",
        )

    @staticmethod
    def _collect_protected_terms(text: str, prompt_terms: list[str]) -> list[str]:
        current = str(text or "")
        terms = []
        for term in sorted({str(item).strip() for item in prompt_terms if str(item).strip()}, key=len, reverse=True):
            if len(term) < 2:
                continue
            if term in current:
                terms.append(term)
        return terms
