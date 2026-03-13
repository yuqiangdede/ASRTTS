from __future__ import annotations

import os
from pathlib import Path


REPO_ID = "Systran/faster-whisper-small"
HF_ENDPOINT = "https://hf-mirror.com"
MODEL_PAGE_URL = f"{HF_ENDPOINT}/{REPO_ID}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "models" / "faster-whisper-small"


def is_non_empty_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def is_downloaded(target: Path) -> bool:
    required = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    return target.is_dir() and all(is_non_empty_file(target / name) for name in required)


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    target = target_root()
    if is_downloaded(target):
        print(f"[skip] {target} already exists")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("未安装 huggingface-hub，请先执行 pip install -r requirements-cpu.txt 或 requirements-gpu.txt") from exc

    print(f"[mirror] {HF_ENDPOINT}")
    print(f"[manual] {MODEL_PAGE_URL}")
    print(f"[download] {REPO_ID} -> {target}")
    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(target),
            resume_download=True,
            endpoint=HF_ENDPOINT,
        )
    except TypeError:
        try:
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(target),
                endpoint=HF_ENDPOINT,
            )
        except Exception as exc:
            print(f"[error] 下载失败: {type(exc).__name__}: {exc}")
            print(f"[hint] 可手工下载后放入目录: {target}")
            print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
            return 1
    except Exception as exc:
        print(f"[error] 下载失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1
    print(f"[done] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
