from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib import parse, request


REPO_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
API_URL = f"https://huggingface.co/api/models/{REPO_ID}/tree/main?recursive=1"
BASE_DOWNLOAD_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
RUNTIME_FILES = {
    "runtime/model.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py",
    "runtime/ctc.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/ctc.py",
    "runtime/decode.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/decode.py",
    "runtime/tools/utils.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/tools/utils.py",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "models" / "Fun-ASR-Nano-2512"


def vad_target_root() -> Path:
    return project_root() / "models" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"


def fetch_tree() -> list[dict]:
    req = request.Request(API_URL, headers={"User-Agent": "ASRTTS-Downloader/1.0"})
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_files(items: list[dict]) -> list[str]:
    files: list[str] = []
    for item in items:
        if str(item.get("type", "")).lower() != "file":
            continue
        path = str(item.get("path", "") or "").strip()
        if not path:
            continue
        files.append(path)
    return files


def should_skip_download(dst: Path) -> bool:
    if not dst.exists() or not dst.is_file():
        return False
    return True


def download_file(rel_path: str) -> None:
    dst = target_root() / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if should_skip_download(dst):
        print(f"[skip] {rel_path} (local exists)")
        return
    encoded_path = "/".join(parse.quote(part) for part in rel_path.split("/"))
    url = f"{BASE_DOWNLOAD_URL}/{encoded_path}?download=true"
    print(f"[download] {rel_path}")
    req = request.Request(url, headers={"User-Agent": "ASRTTS-Downloader/1.0"})
    with request.urlopen(req, timeout=600) as resp, dst.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def download_runtime_file(rel_path: str, url: str) -> None:
    dst = target_root() / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if should_skip_download(dst):
        print(f"[skip] {rel_path} (local exists)")
        return
    print(f"[download] {rel_path}")
    req = request.Request(url, headers={"User-Agent": "ASRTTS-Downloader/1.0"})
    with request.urlopen(req, timeout=120) as resp, dst.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def ensure_text_file(rel_path: str, content: str) -> None:
    dst = target_root() / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if should_skip_download(dst):
        print(f"[skip] {rel_path} (local exists)")
        return
    print(f"[write] {rel_path}")
    dst.write_text(content, encoding="utf-8")


def copy_vad_from_cache(target: Path) -> bool:
    if not VAD_CACHE_DIR.is_dir():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(VAD_CACHE_DIR, target, dirs_exist_ok=True)
    return True


def download_vad_model() -> None:
    target = vad_target_root()
    if target.is_dir():
        print(f"[skip] {target} already exists")
        return

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("未安装 modelscope，请先执行 pip install -r requirements.txt") from exc

    os.environ.setdefault("MODELSCOPE_CACHE", str(project_root() / ".modelscope_cache"))
    try:
        print(f"[download] {VAD_MODEL_ID} -> {target}")
        snapshot_download(VAD_MODEL_ID, local_dir=str(target))
    except TypeError:
        print("[fallback] snapshot_download 不支持 local_dir，改为先下载到缓存再复制到项目目录")
        snapshot_download(VAD_MODEL_ID)
        if not copy_vad_from_cache(target):
            raise RuntimeError("VAD 模型已下载，但未能从缓存复制到项目目录。")

    if not target.is_dir():
        if not copy_vad_from_cache(target):
            raise RuntimeError("VAD 模型下载完成，但项目目录中未找到结果。")

    print(f"[done] {target}")


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    root = target_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[target] {root}")
    tree = fetch_tree()
    files = iter_files(tree)
    if not files:
        raise RuntimeError("未从 Hugging Face 获取到文件清单。")
    print(f"[files] {len(files)}")
    for rel_path in files:
        download_file(rel_path)
    for rel_path, url in RUNTIME_FILES.items():
        download_runtime_file(rel_path, url)
    ensure_text_file("runtime/tools/__init__.py", "")
    download_vad_model()
    print("[done] Fun-ASR-Nano-2512 下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
