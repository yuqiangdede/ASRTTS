from __future__ import annotations

import os
import shutil
from pathlib import Path


MODEL_ID = "manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
MODEL_PAGE_URL = "https://modelscope.cn/models/manyeyes/paraformer-seaco-large-zh-timestamp-int8-onnx-offline/files"
CACHE_DIR = Path.home() / ".cache" / "modelscope" / "hub" / "models" / "manyeyes" / "paraformer-seaco-large-zh-timestamp-int8-onnx-offline"
REQUIRED_FILES = [
    "am.mvn",
    "config.yaml",
    "configuration.json",
    "tokens.json",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "models" / "paraformer-seaco-large-zh-timestamp-int8-onnx-offline"


def is_non_empty_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def find_onnx_files(target: Path) -> list[Path]:
    if not target.is_dir():
        return []
    return [item for item in target.rglob("*.onnx") if is_non_empty_file(item)]


def print_required_file_status(target: Path) -> list[str]:
    missing: list[str] = []
    for relative_path in REQUIRED_FILES:
        file_path = target / relative_path
        if is_non_empty_file(file_path):
            print(f"[check] ok: {relative_path}")
        else:
            print(f"[check] missing: {relative_path}")
            missing.append(relative_path)

    onnx_files = find_onnx_files(target)
    if onnx_files:
        print(f"[check] ok: *.onnx ({onnx_files[0].relative_to(target)})")
    else:
        print("[check] missing: *.onnx")
        missing.append("*.onnx")
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
        raise RuntimeError("未安装 modelscope，请先执行 pip install -r requirements-cpu.txt 或 requirements-gpu.txt") from exc

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
            print("[error] Paraformer ONNX 模型已下载，但未能从缓存复制到项目目录。")
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
            print("[error] Paraformer ONNX 模型下载完成，但项目目录中未找到结果。")
            print(f"[hint] 可手工下载后放入目录: {target}")
            print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
            return 1

    missing = print_required_file_status(target)
    if missing:
        print(f"[error] Paraformer ONNX 模型目录仍缺少必需文件: {', '.join(missing)}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1

    print(f"[done] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
