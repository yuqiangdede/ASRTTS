from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib import parse, request


REPO_ID = "FunAudioLLM/Fun-ASR-Nano-2512"
HF_ENDPOINT = "https://hf-mirror.com"
API_URL = f"{HF_ENDPOINT}/api/models/{REPO_ID}/tree/main?recursive=1"
BASE_DOWNLOAD_URL = f"{HF_ENDPOINT}/{REPO_ID}/resolve/main"
MODEL_PAGE_URL = f"{HF_ENDPOINT}/{REPO_ID}"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
VAD_MODEL_PAGE_URL = f"{HF_ENDPOINT}/{VAD_MODEL_ID}"
VAD_CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
RUNTIME_FILES = {
    "runtime/model.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py",
    "runtime/ctc.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/ctc.py",
    "runtime/decode.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/decode.py",
    "runtime/tools/utils.py": "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/tools/utils.py",
}
REQUIRED_MODEL_FILES = [
    "config.yaml",
    "model.pt",
    "Qwen3-0.6B/config.json",
]


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


def is_non_empty_file(dst: Path) -> bool:
    if not dst.exists() or not dst.is_file():
        return False
    try:
        return dst.stat().st_size > 0
    except OSError:
        return False


def should_skip_download(dst: Path) -> bool:
    return is_non_empty_file(dst)


def print_required_file_status(root: Path) -> list[str]:
    missing: list[str] = []
    for rel_path in REQUIRED_MODEL_FILES:
        file_path = root / rel_path
        if is_non_empty_file(file_path):
            print(f"[check] ok: {rel_path}")
        else:
            print(f"[check] missing_or_empty: {rel_path}")
            missing.append(rel_path)
    return missing


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
    missing = print_vad_required_file_status(target)
    if target.is_dir() and not missing:
        print(f"[skip] {target} already exists")
        return
    if target.is_dir() and missing:
        print(f"[check] {target} 缺少必需文件，将继续下载/补齐: {', '.join(missing)}")

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

    missing = print_vad_required_file_status(target)
    if missing:
        raise RuntimeError(f"VAD 模型目录仍缺少必需文件：{', '.join(missing)}")

    print(f"[done] {target}")


def print_vad_required_file_status(target: Path) -> list[str]:
    required = ["am.mvn", "config.yaml", "configuration.json", "model.pt"]
    missing: list[str] = []
    for relative_path in required:
        file_path = target / relative_path
        if is_non_empty_file(file_path):
            print(f"[check] ok: {relative_path}")
        else:
            print(f"[check] missing: {relative_path}")
            missing.append(relative_path)
    return missing


def main() -> int:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    root = target_root()
    root.mkdir(parents=True, exist_ok=True)
    print(f"[mirror] {HF_ENDPOINT}")
    print(f"[manual] FunASR: {MODEL_PAGE_URL}")
    print(f"[manual] VAD: {VAD_MODEL_PAGE_URL}")
    print(f"[target] {root}")
    missing = print_required_file_status(root)
    if missing:
        print(f"[check] 主模型存在缺失或空文件，将继续下载/补齐: {', '.join(missing)}")
    try:
        tree = fetch_tree()
    except Exception as exc:
        print(f"[error] 获取文件清单失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工访问模型页面: {MODEL_PAGE_URL}")
        return 1
    files = iter_files(tree)
    if not files:
        print("[error] 未从 Hugging Face 获取到文件清单。")
        print(f"[hint] 可手工访问模型页面: {MODEL_PAGE_URL}")
        return 1
    print(f"[files] {len(files)}")
    try:
        for rel_path in files:
            download_file(rel_path)
        for rel_path, url in RUNTIME_FILES.items():
            download_runtime_file(rel_path, url)
    except Exception as exc:
        print(f"[error] FunASR 主模型下载失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工下载后放入目录: {root}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1
    ensure_text_file("runtime/tools/__init__.py", "")
    missing = print_required_file_status(root)
    if missing:
        print(f"[error] FunASR 主模型目录仍缺少或为空文件: {', '.join(missing)}")
        print(f"[hint] 可手工下载后放入目录: {root}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1
    try:
        download_vad_model()
    except Exception as exc:
        print(f"[error] VAD 模型下载失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工下载后放入目录: {vad_target_root()}")
        print(f"[hint] 模型页面: {VAD_MODEL_PAGE_URL}")
        return 1
    print("[done] Fun-ASR-Nano-2512 下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
