from __future__ import annotations

import os
from pathlib import Path


REPO_ID = "Systran/faster-whisper-small"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "faster-whisper-small"


def is_downloaded(target: Path) -> bool:
    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    return target.is_dir() and all((target / name).exists() for name in required)


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    target = target_root()
    if is_downloaded(target):
        print(f"[skip] {target} already exists")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("未安装 huggingface-hub，请先执行 pip install -r requirements.txt") from exc

    print(f"[download] {REPO_ID} -> {target}")
    try:
        snapshot_download(repo_id=REPO_ID, local_dir=str(target), resume_download=True)
    except TypeError:
        snapshot_download(repo_id=REPO_ID, local_dir=str(target))
    print(f"[done] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
