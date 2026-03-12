from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DomainProfile:
    name: str
    prompt_terms: list[str]
    phrase_rules: list[dict[str, Any]]
    confusion_rules: list[dict[str, Any]]


class DomainConfigLoader:
    def __init__(self, config_path: str | Path, common_terms_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path).resolve()
        self.common_terms_path = Path(common_terms_path).resolve() if common_terms_path else None
        self._common_prompt_terms = self._load_common_prompt_terms()
        self._profiles = self._load_profiles()

    def _load_common_prompt_terms(self) -> list[str]:
        if self.common_terms_path is None or not self.common_terms_path.is_file():
            return []
        data = yaml.safe_load(self.common_terms_path.read_text(encoding="utf-8")) or {}
        terms = data.get("prompt_terms", []) if isinstance(data, dict) else []
        return [str(item).strip() for item in terms if str(item).strip()]

    def _load_profiles(self) -> dict[str, DomainProfile]:
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        domains = data.get("domains", {}) if isinstance(data, dict) else {}
        profiles: dict[str, DomainProfile] = {}

        for name, raw in domains.items():
            if not isinstance(raw, dict):
                continue
            base_terms = [str(item).strip() for item in raw.get("prompt_terms", []) if str(item).strip()]
            merged_terms: list[str] = []
            seen: set[str] = set()
            for term in [*base_terms, *self._common_prompt_terms]:
                if term in seen:
                    continue
                seen.add(term)
                merged_terms.append(term)
            profiles[str(name)] = DomainProfile(
                name=str(name),
                prompt_terms=merged_terms,
                phrase_rules=[rule for rule in raw.get("phrase_rules", []) if isinstance(rule, dict)],
                confusion_rules=[rule for rule in raw.get("confusion_rules", []) if isinstance(rule, dict)],
            )

        if not profiles:
            raise RuntimeError(f"未在 {self.config_path} 中找到有效的业务域配置。")

        return profiles

    def list_domains(self) -> list[str]:
        return sorted(self._profiles.keys())

    def get_profile(self, domain: str) -> DomainProfile:
        key = str(domain or "").strip()
        if key not in self._profiles:
            raise KeyError(f"未知业务域：{domain}")
        return self._profiles[key]

    def build_initial_prompt(self, domain: str) -> str:
        profile = self.get_profile(domain)
        if not profile.prompt_terms:
            return ""
        return "，".join(profile.prompt_terms)
