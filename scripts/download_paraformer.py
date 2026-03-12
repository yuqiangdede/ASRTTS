from __future__ import annotations

import os
import shutil
from pathlib import Path


MODEL_ID = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
MODEL_PAGE_URL = "https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files"
CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "iic" / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
REQUIRED_FILES = [
    "am.mvn",
    "config.yaml",
    "configuration.json",
    "model.pt",
    "seg_dict",
    "tokens.json",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "models" / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def is_non_empty_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def print_required_file_status(target: Path) -> list[str]:
    missing: list[str] = []
    for relative_path in REQUIRED_FILES:
        file_path = target / relative_path
        if is_non_empty_file(file_path):
            print(f"[check] ok: {relative_path}")
        else:
            print(f"[check] missing: {relative_path}")
            missing.append(relative_path)
    return missing


def copy_from_cache(target: Path) -> bool:
    if not CACHE_DIR.is_dir():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CACHE_DIR, target, dirs_exist_ok=True)
    return True


def main() -> int:
    target = target_root()
    print("[source] ModelScope")
    print(f"[manual] {MODEL_PAGE_URL}")
    print(f"[target] {target}")

    missing = print_required_file_status(target)
    if target.is_dir() and not missing:
        print(f"[skip] {target} already exists")
        return 0
    if target.is_dir() and missing:
        print(f"[check] {target} 缺少必需文件，将继续下载/补齐: {', '.join(missing)}")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("未安装 modelscope，请先执行 pip install -r requirements.txt") from exc

    os.environ.setdefault("MODELSCOPE_CACHE", str(project_root() / ".modelscope_cache"))
    try:
        print(f"[download] {MODEL_ID} -> {target}")
        snapshot_download(MODEL_ID, local_dir=str(target))
    except TypeError:
        print("[fallback] snapshot_download 不支持 local_dir，改为先下载到缓存再复制到项目目录")
        try:
            snapshot_download(MODEL_ID)
        except Exception as exc:  # noqa: BLE001
            print(f"[error] 下载失败: {type(exc).__name__}: {exc}")
            print(f"[hint] 可手工下载后放入目录: {target}")
            print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
            return 1
        if not copy_from_cache(target):
            print("[error] Paraformer 模型已下载，但未能从缓存复制到项目目录。")
            print(f"[hint] 可手工下载后放入目录: {target}")
            print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 下载失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1

    if not target.is_dir():
        if not copy_from_cache(target):
            print("[error] Paraformer 模型下载完成，但项目目录中未找到结果。")
            print(f"[hint] 可手工下载后放入目录: {target}")
            print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
            return 1

    missing = print_required_file_status(target)
    if missing:
        print(f"[error] Paraformer 模型目录仍缺少必需文件: {', '.join(missing)}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1

    print(f"[done] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
