from __future__ import annotations

import os
import socket
import time
import urllib.request
from pathlib import Path

REPO_ID = "dropbox-dash/faster-whisper-large-v3-turbo"
HF_ENDPOINT = "https://hf-mirror.com"
MODEL_PAGE_URL = f"{HF_ENDPOINT}/{REPO_ID}"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "faster-whisper-large-v3-turbo"


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


def check_dns(host: str) -> None:
    ip = socket.gethostbyname(host)
    print(f"[dns] {host} -> {ip}")


def check_http(url: str, timeout: int = 15) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        print(f"[http] {url} -> {resp.status}")


def precheck() -> None:
    print("[precheck] start")
    check_dns("hf-mirror.com")
    check_http(f"{HF_ENDPOINT}/api/models/{REPO_ID}")
    print("[precheck] ok")


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    target = target_root()

    if is_downloaded(target):
        print(f"[skip] {target} already exists")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "未安装 huggingface-hub，请先执行 pip install -r requirements.txt"
        ) from exc

    print(f"[mirror] {HF_ENDPOINT}")
    print(f"[manual] {MODEL_PAGE_URL}")
    print(f"[download] {REPO_ID} -> {target}")
    target.mkdir(parents=True, exist_ok=True)

    try:
        precheck()
    except Exception as exc:
        print(f"[error] 网络预检查失败: {type(exc).__name__}: {exc}")
        print("[hint] 重点检查 hf-mirror.com / 企业DNS / 内网策略")
        print(f"[hint] 也可手工访问: {MODEL_PAGE_URL}")
        return 2

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            print(f"[try] attempt {attempt}/3")
            snapshot_download(
                repo_id=REPO_ID,
                local_dir=str(target),
                local_dir_use_symlinks=False,
                endpoint=HF_ENDPOINT,
            )
            print(f"[done] {target}")
            return 0
        except Exception as exc:
            last_error = exc
            print(f"[warn] 下载失败，第 {attempt} 次: {type(exc).__name__}: {exc}")
            if attempt < 3:
                time.sleep(3)

    print(f"[error] 下载最终失败: {type(last_error).__name__}: {last_error}")
    print(f"[hint] 可手工下载后放入目录: {target}")
    print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
