"""Output container assembly (.cbz, .pdf, .zip, .cbr, .rar, single image) and partial resume tracking."""

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from typing import List

try:
    from PIL import Image
except ImportError:
    Image = None

from ..config import SUPPORTED_IMAGE_EXTS


def get_partial_filepath(final_output: str) -> str:
    """Generates standard .partial output path (e.g. manga.en.partial.cbz)."""
    base, ext = os.path.splitext(final_output)
    return f"{base}.partial{ext}"


def pack_output_pages(page_paths: List[str], output_path: str, format_ext: str) -> str:
    """Packages processed pages into destination container (.cbz, .pdf, .zip, single image, etc.)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fmt = format_ext.lower()

    if fmt in SUPPORTED_IMAGE_EXTS:
        if page_paths:
            if fmt == ".webp":
                shutil.copyfile(page_paths[0], output_path)
            elif Image is not None:
                _pil_fmt_map = {
                    ".png": "PNG",
                    ".jpg": "JPEG",
                    ".jpeg": "JPEG",
                    ".bmp": "BMP",
                    ".tiff": "TIFF",
                    ".tif": "TIFF",
                }
                save_fmt = _pil_fmt_map.get(fmt, "PNG")
                img = Image.open(page_paths[0]).convert("RGB")
                img.save(output_path, save_fmt)
            else:
                shutil.copyfile(page_paths[0], output_path)
        return output_path

    if fmt in (".cbz", ".zip"):
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, p in enumerate(page_paths):
                arcname = f"{idx + 1:04d}{os.path.splitext(p)[1]}"
                zf.write(p, arcname=arcname)
        return output_path

    elif fmt == ".pdf":
        images = []
        for p in page_paths:
            if Image is not None:
                im = Image.open(p).convert("RGB")
                images.append(im)
        if images:
            images[0].save(output_path, "PDF", resolution=150.0, save_all=True, append_images=images[1:])
        return output_path

    elif fmt in (".tar", ".cbt"):
        with tarfile.open(output_path, "w") as tf:
            for idx, p in enumerate(page_paths):
                arcname = f"{idx + 1:04d}{os.path.splitext(p)[1]}"
                tf.add(p, arcname=arcname)
        return output_path

    elif fmt in (".cbr", ".rar"):
        rar_bin = shutil.which("rar")
        if not rar_bin and sys.platform == "win32":
            for cand in [
                r"C:\Program Files\WinRAR\Rar.exe",
                r"C:\Program Files (x86)\WinRAR\Rar.exe",
            ]:
                if os.path.isfile(cand):
                    rar_bin = cand
                    break

        if rar_bin:
            if os.path.exists(output_path):
                os.remove(output_path)
            subprocess.run([rar_bin, "a", "-inul", "-ep", output_path] + page_paths, check=True)
            return output_path
        else:
            raise RuntimeError(
                "Packaging as .cbr/.rar requires the 'rar' command-line compression tool.\n"
                "  - Linux: sudo apt install rar\n"
                "  - macOS: brew install rar\n"
                "  - Windows: Install WinRAR (adds Rar.exe)\n"
                "Tip: .cbz (Comic Book ZIP) is the universal open standard and works everywhere out-of-the-box."
            )

    else:
        os.makedirs(output_path, exist_ok=True)
        for idx, p in enumerate(page_paths):
            dest = os.path.join(output_path, f"{idx + 1:04d}{os.path.splitext(p)[1]}")
            shutil.copyfile(p, dest)
        return output_path
