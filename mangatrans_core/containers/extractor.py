"""Archive and document extractor with Zip-Slip path sanitization."""

import os
import shutil
import subprocess
import tarfile
import zipfile
from typing import List

from ..config import SUPPORTED_IMAGE_EXTS, natural_sort_key


def is_safe_path(base_dir: str, path: str, follow_symlinks: bool = True) -> bool:
    """Guarantees path does not escape base_dir (Zip-Slip vulnerability defense)."""
    if follow_symlinks:
        match_path = os.path.realpath(path)
    else:
        match_path = os.path.abspath(path)
    return os.path.commonpath([base_dir, match_path]) == os.path.abspath(base_dir)


def extract_input_pages(input_path: str, extract_dir: str) -> List[str]:
    """Extracts all pages from archive/PDF/folder with security validation."""
    if os.path.isdir(input_path):
        pages = []
        for root, _, files in os.walk(input_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    pages.append(os.path.join(root, f))
        return sorted(pages, key=natural_sort_key)

    ext = os.path.splitext(input_path)[1].lower()

    if ext in (".cbz", ".zip"):
        with zipfile.ZipFile(input_path, "r") as zf:
            extracted = []
            for n in sorted(zf.namelist(), key=natural_sort_key):
                if os.path.splitext(n)[1].lower() in SUPPORTED_IMAGE_EXTS and not n.startswith("__MACOSX"):
                    out_p = os.path.join(extract_dir, os.path.basename(n))
                    with open(out_p, "wb") as f:
                        f.write(zf.read(n))
                    extracted.append(out_p)
            return extracted

    elif ext in (".cbr", ".rar"):
        if shutil.which("unrar"):
            subprocess.run(["unrar", "e", "-inul", "-y", input_path, extract_dir], check=True)
        elif shutil.which("7z"):
            subprocess.run(["7z", "e", f"-o{extract_dir}", "-y", input_path], capture_output=True, check=True)
        else:
            raise RuntimeError("Extracting .cbr/.rar requires unrar or 7z. Install: sudo apt install unrar")
        pages = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
        ]
        return sorted(pages, key=natural_sort_key)

    elif ext in (".tar", ".cbt", ".cb7", ".7z"):
        if ext in (".tar", ".cbt"):
            with tarfile.open(input_path, "r:*") as tf:
                for member in tf.getmembers():
                    if is_safe_path(extract_dir, member.name):
                        try:
                            tf.extract(member, path=extract_dir, filter="data")
                        except TypeError:
                            tf.extract(member, path=extract_dir)
        else:
            if shutil.which("7z"):
                subprocess.run(["7z", "e", f"-o{extract_dir}", "-y", input_path], capture_output=True, check=True)
            else:
                raise RuntimeError("Extracting .7z/.cb7 requires 7z. Install: sudo apt install p7zip-full")
        pages = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
        ]
        return sorted(pages, key=natural_sort_key)

    elif ext == ".pdf":
        if shutil.which("pdftoppm"):
            subprocess.run(["pdftoppm", "-png", "-r", "150", input_path, os.path.join(extract_dir, "page")], check=True)
        elif shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", input_path, os.path.join(extract_dir, "page_%04d.png")], check=True)
        else:
            raise RuntimeError("Extracting PDF pages requires pdftoppm or ffmpeg.")
        pages = [
            os.path.join(extract_dir, f)
            for f in os.listdir(extract_dir)
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTS
        ]
        return sorted(pages, key=natural_sort_key)

    elif ext in SUPPORTED_IMAGE_EXTS:
        return [input_path]

    raise ValueError(f"Unsupported input file format: {ext}")
