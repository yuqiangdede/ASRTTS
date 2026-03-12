from __future__ import annotations

from typing import Any


class PhraseCorrector:
    def apply(self, text: str, rules: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        current = str(text or "")
        applied: list[dict[str, Any]] = []

        for rule in rules:
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
