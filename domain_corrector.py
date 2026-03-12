from __future__ import annotations

from typing import Any


class DomainCorrector:
    def apply(self, text: str, rules: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        current = str(text or "")
        applied: list[dict[str, Any]] = []

        for rule in rules:
            current, per_rule_applied = self._apply_single_rule(current, rule)
            applied.extend(per_rule_applied)

        return current, applied

    def _normalize_weight_map(self, value: Any, *, default_weight: float = 1.0) -> dict[str, float]:
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items() if str(k).strip()}
        if isinstance(value, list):
            return {str(item): float(default_weight) for item in value if str(item).strip()}
        return {}

    def _score_context(self, context: str, rule: dict[str, Any]) -> tuple[float, list[str], list[str]]:
        positive_map = self._normalize_weight_map(rule.get("context_keywords"), default_weight=1.0)
        negative_map = self._normalize_weight_map(rule.get("negative_keywords"), default_weight=1.0)

        score = 0.0
        positive_hits: list[str] = []
        negative_hits: list[str] = []

        for keyword, weight in positive_map.items():
            if keyword in context:
                score += weight
                positive_hits.append(keyword)

        for keyword, weight in negative_map.items():
            if keyword in context:
                score -= weight
                negative_hits.append(keyword)

        return score, positive_hits, negative_hits

    def _apply_single_rule(self, text: str, rule: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
        current = str(text or "")
        applied: list[dict[str, Any]] = []

        source_terms = [str(item).strip() for item in rule.get("source_terms", []) if str(item).strip()]
        target_term = str(rule.get("target_term", "") or "").strip()
        rule_name = str(rule.get("name", target_term or "confusion_rule") or "confusion_rule")
        threshold = float(rule.get("threshold", 1.0) or 1.0)
        window = int(rule.get("window", 6) or 6)

        if not source_terms or not target_term:
            return current, applied

        for source in sorted(source_terms, key=len, reverse=True):
            cursor = 0
            while cursor < len(current):
                index = current.find(source, cursor)
                if index < 0:
                    break

                left = max(0, index - window)
                right = min(len(current), index + len(source) + window)
                context = current[left:right]
                score, positive_hits, negative_hits = self._score_context(context, rule)

                if score >= threshold:
                    before = current
                    current = current[:index] + target_term + current[index + len(source) :]
                    applied.append(
                        {
                            "stage": "confusion",
                            "rule": rule_name,
                            "pattern": source,
                            "replacement": target_term,
                            "score": score,
                            "threshold": threshold,
                            "positive_hits": positive_hits,
                            "negative_hits": negative_hits,
                            "before": before,
                            "after": current,
                        }
                    )
                    cursor = index + len(target_term)
                else:
                    cursor = index + len(source)

        return current, applied
