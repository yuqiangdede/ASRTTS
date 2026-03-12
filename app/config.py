from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path() -> Path:
    return project_root() / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise RuntimeError(f"缺少配置文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(config: dict[str, Any]) -> Path:
    path = config_path()
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()
