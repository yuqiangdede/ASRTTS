from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path
from urllib import request


MODEL_NAME = "sherpa-onnx-zipformer-zh-en-2023-11-22"
MODEL_ARCHIVE = f"{MODEL_NAME}.tar.bz2"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_ARCHIVE}"
MODEL_PAGE_URL = "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html"
REQUIRED_PATTERNS = [
    "tokens.txt",
    "encoder-*.onnx",
    "decoder-*.onnx",
    "joiner-*.onnx",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def target_root() -> Path:
    return project_root() / "models" / MODEL_NAME


def archive_path() -> Path:
    return project_root() / "models" / MODEL_ARCHIVE


def resolve_manual_source(argv: list[str]) -> Path | None:
    if len(argv) < 2:
        return None
    value = str(argv[1] or "").strip().strip('"')
    if not value:
        return None
    return Path(value).expanduser().resolve()


def is_non_empty_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def print_required_file_status(target: Path) -> list[str]:
    missing: list[str] = []
    for relative_path in REQUIRED_PATTERNS:
        if "*" in relative_path:
            matched = [item for item in target.glob(relative_path) if is_non_empty_file(item)]
            if matched:
                print(f"[check] ok: {relative_path} ({matched[0].name})")
            else:
                print(f"[check] missing: {relative_path}")
                missing.append(relative_path)
            continue

        file_path = target / relative_path
        if is_non_empty_file(file_path):
            print(f"[check] ok: {relative_path}")
        else:
            print(f"[check] missing: {relative_path}")
            missing.append(relative_path)
    return missing


def download_archive(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if is_non_empty_file(dst):
        print(f"[skip] {dst.name} (local exists)")
        return
    print(f"[download] {MODEL_URL}")
    req = request.Request(MODEL_URL, headers={"User-Agent": "ASRTTS-Downloader/1.0"})
    with request.urlopen(req, timeout=600) as resp, dst.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def extract_archive(src: Path, target: Path) -> None:
    temp_extract_root = target.parent / f".extract_{MODEL_NAME}"
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root, ignore_errors=True)
    temp_extract_root.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {src.name} -> {target.parent}")
    with tarfile.open(src, "r:bz2") as tar:
        tar.extractall(path=temp_extract_root)
    extracted = temp_extract_root / MODEL_NAME
    if not extracted.is_dir():
        raise RuntimeError(f"压缩包内未找到目录：{MODEL_NAME}")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(extracted), str(target))
    shutil.rmtree(temp_extract_root, ignore_errors=True)


def copy_from_directory(src: Path, target: Path) -> None:
    print(f"[copy] {src} -> {target}")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(src, target, dirs_exist_ok=True)


def prepare_from_manual_source(source: Path, target: Path, archive: Path) -> None:
    if source.is_dir():
        copy_from_directory(source, target)
        return

    if source.is_file():
        suffixes = "".join(source.suffixes).lower()
        if suffixes.endswith(".tar.bz2"):
            archive.parent.mkdir(parents=True, exist_ok=True)
            if source != archive:
                print(f"[copy] {source} -> {archive}")
                shutil.copy2(source, archive)
            extract_archive(archive, target)
            return

    raise RuntimeError(
        f"不支持的本地来源：{source}。请传入已解压的模型目录，或 {MODEL_ARCHIVE} 压缩包。"
    )


def main() -> int:
    target = target_root()
    archive = archive_path()
    manual_source = resolve_manual_source(sys.argv)
    if manual_source is not None:
        print(f"[source] local: {manual_source}")
    else:
        print("[source] GitHub Releases")
    print(f"[manual] {MODEL_PAGE_URL}")
    print(f"[target] {target}")

    missing = print_required_file_status(target)
    if target.is_dir() and not missing:
        print(f"[skip] {target} already exists")
        return 0
    if target.is_dir() and missing:
        print(f"[check] {target} 缺少必需文件，将继续下载/补齐: {', '.join(missing)}")

    try:
        if manual_source is not None:
            prepare_from_manual_source(manual_source, target, archive)
        else:
            download_archive(archive)
            extract_archive(archive, target)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] 下载或解压失败: {type(exc).__name__}: {exc}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1

    missing = print_required_file_status(target)
    if missing:
        print(f"[error] Sherpa ONNX 模型目录仍缺少必需文件: {', '.join(missing)}")
        print(f"[hint] 可手工下载后放入目录: {target}")
        print(f"[hint] 模型页面: {MODEL_PAGE_URL}")
        return 1

    print(f"[done] {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
