from __future__ import annotations

import re
from typing import Any


class PhraseCorrector:
    def apply(self, text: str, rules: list[dict[str, Any]], *, normalize_spacing: bool = True) -> tuple[str, list[dict[str, Any]]]:
        current = str(text or "")
        applied: list[dict[str, Any]] = []

        normalized = self.normalize_text(current) if normalize_spacing else current
        if normalize_spacing and normalized != current:
            applied.append(
                {
                    "stage": "phrase",
                    "rule": "normalize_asr_spacing",
                    "pattern": current,
                    "replacement": normalized,
                    "before": current,
                    "after": normalized,
                }
            )
            current = normalized

        for rule in self._sort_rules_for_matching(rules):
            replacement = str(rule.get("replacement", "") or "").strip()
            patterns = [str(item).strip() for item in rule.get("patterns", []) if str(item).strip()]
            rule_name = str(rule.get("name", replacement or "phrase_rule") or "phrase_rule")
            if not replacement or not patterns:
                continue

            for pattern in sorted(patterns, key=len, reverse=True):
                if pattern not in current:
                    continue
                before = current
                current = current.replace(pattern, replacement)
                applied.append(
                    {
                        "stage": "phrase",
                        "rule": rule_name,
                        "pattern": pattern,
                        "replacement": replacement,
                        "before": before,
                        "after": current,
                    }
                )

        return current, applied

    @staticmethod
    def _sort_rules_for_matching(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        indexed_rules = list(enumerate(rules))

        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            index, rule = item
            patterns = [str(value).strip() for value in rule.get("patterns", []) if str(value).strip()]
            max_len = max((len(pattern) for pattern in patterns), default=0)
            return (-max_len, index)

        return [rule for _, rule in sorted(indexed_rules, key=sort_key)]

    @staticmethod
    def normalize_text(text: str) -> str:
        current = str(text or "")
        if not current or " " not in current:
            return current

        cjk_or_digit = r"\u4e00-\u9fff0-9"
        zh_punct = r"，。！？；：、（）《》【】“”‘’"

        patterns = [
            rf"(?<=[{cjk_or_digit}])\s+(?=[{cjk_or_digit}])",
            rf"(?<=[{cjk_or_digit}])\s+(?=[{zh_punct}])",
            rf"(?<=[{zh_punct}])\s+(?=[{cjk_or_digit}])",
        ]

        normalized = current
        for pattern in patterns:
            normalized = re.sub(pattern, "", normalized)
        normalized = re.sub(r"\s{2,}", " ", normalized).strip()
        return normalized
