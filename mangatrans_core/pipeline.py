"""Full page rendering, batch translation pipeline, overlapped prefetching, and resume management."""

import concurrent.futures
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from .ai import NarrativeMemory, query_page_translation
from .config import (
    LiveProgress,
    SUPPORTED_ARCHIVE_EXTS,
    SUPPORTED_DOC_EXTS,
    SUPPORTED_IMAGE_EXTS,
    _bold,
    _cyan,
    _dim,
    _green,
    _red,
    _yellow,
    get_prog_name,
    mask_key,
    natural_sort_key,
    parse_page_selection,
    print_banner,
    resolve_language_info,
)
from .containers import extract_input_pages, get_partial_filepath, pack_output_pages
from .keys import (
    AllKeysExhaustedError,
    DynamicKeyManager,
    format_duration,
    load_key_cooldowns,
)
from .typesetting import (
    _measure_or_draw,
    draw_fitted,
    draw_outlined,
    get_font_paths,
    split_text_for_regions,
)
from .vision import inpaint_bubble, inpaint_floating, is_lobe_viable_for_text, partition_overlapping_regions


def is_sfx_item(item: Dict[str, Any]) -> bool:
    """Detects whether an item is an onomatopoeia / sound effect that should be preserved as raw art."""
    b_type = str(item.get("bubble_type", "")).lower().strip()
    if b_type == "sfx":
        return True
    c_type = str(item.get("container_type", "")).lower().strip()
    trans = str(item.get("translated_text", "")).strip()
    orig = str(item.get("original_text", "")).strip()

    if c_type == "floating" and (trans.startswith("*") and trans.endswith("*")):
        return True

    sfx_orig_keywords = {
        "あせ", "アセ", "汗", "ドキッ", "ドキドキ", "ハァ", "ハァハァ", "ため息", "ビクッ", "ビクン",
        "ピクッ", "ピクン", "コクリ", "ゴクリ", "ぎゅっ", "ギュッ", "ふぅ", "フゥ", "はぁ", "ハァ",
        "あっ", "アッ", "んっ", "ンッ", "うっ", "ウッ", "ちゅ", "チュ", "じーっ", "チラッ"
    }

    sfx_keywords = {
        "squish", "thump", "plump", "phew", "twitch", "sigh", "gasp", "pant", "sweat", "ase",
        "thump thump", "doki doki", "heartbeat", "rustle", "slide", "creak", "bang", "boom",
        "splat", "squeeze", "rub", "stroke", "pant pant", "haa haa", "haah", "kya", "kyaa",
        "gasp", "gulp", "smack", "flutter", "stare", "glance", "nod", "snicker", "chuckle"
    }
    clean_trans = trans.strip("*~!. ").lower()
    clean_orig = orig.strip("*~!. \n")
    if c_type == "floating" and (clean_trans in sfx_keywords or clean_orig in sfx_orig_keywords):
        return True

    # Spatial / area filter: loose floating particle annotations (e.g. sweat marks)
    box = item.get("box_2d")
    if c_type == "floating" and box and len(box) == 4:
        bw = abs(box[3] - box[1])
        bh = abs(box[2] - box[0])
        words_count = len(clean_trans.split())
        if words_count <= 2 and (bw * bh < 900 or (bw <= 28 and bh <= 28)):
            return True

    return False


def is_actual_caption_box(img_bgr: "np.ndarray", xmin: int, ymin: int, xmax: int, ymax: int) -> bool:
    """Checks whether a region is genuinely a framed rectangular caption box vs unboxed monologue."""
    if cv2 is None or np is None:
        return False
    if len(img_bgr.shape) == 2:
        gray = img_bgr
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 1:
        gray = img_bgr[:, :, 0]
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    roi = gray[ymin:ymax, xmin:xmax]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return False

    b_band = max(2, min(5, int(min(roi.shape) * 0.05)))
    top_dark = np.mean(roi[:b_band, :] < 110) > 0.30
    bot_dark = np.mean(roi[-b_band:, :] < 110) > 0.30
    left_dark = np.mean(roi[:, :b_band] < 110) > 0.30
    right_dark = np.mean(roi[:, -b_band:] < 110) > 0.30
    return (top_dark and bot_dark) or (left_dark and right_dark and (top_dark or bot_dark))


def detect_panel_line_bounds(gray: "np.ndarray", xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
    """Detects continuous straight panel divider lines surrounding a speech bubble in all 4 directions."""
    H, W = gray.shape[:2]
    bw = xmax - xmin
    bh = ymax - ymin
    pad_y = max(110, int(bh * 1.5))
    pad_x = max(40, int(bw * 0.8))
    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2

    # Top panel divider line: start scanning from near top edge inside the box upwards
    scan_y_top = max(0, min(cy, ymin + min(15, max(1, bh // 3))))
    limit_y1 = max(0, ymin - pad_y)
    for y in range(scan_y_top, limit_y1 - 1, -1):
        strip = gray[y, max(0, xmin - 5):min(W, xmax + 5)]
        if len(strip) > 0 and np.mean(strip < 80) >= 0.70:
            limit_y1 = y + 1
            break

    # Bottom panel divider line: start scanning from near bottom edge inside the box downwards
    scan_y_bot = min(H - 1, max(cy, ymax - min(15, max(1, bh // 3))))
    limit_y2 = min(H - 1, ymax + pad_y)
    for y in range(scan_y_bot, limit_y2 + 1):
        strip = gray[y, max(0, xmin - 5):min(W, xmax + 5)]
        if len(strip) > 0 and np.mean(strip < 80) >= 0.70:
            limit_y2 = y - 1
            break

    # Left vertical panel divider line: start scanning from near left edge inside the box leftwards
    scan_x_left = max(0, min(cx, xmin + min(15, max(1, bw // 3))))
    limit_x1 = max(0, xmin - pad_x)
    for x in range(scan_x_left, limit_x1 - 1, -1):
        strip = gray[max(0, ymin - 5):min(H, ymax + 5), x]
        if len(strip) > 0 and np.mean(strip < 80) >= 0.70:
            limit_x1 = x + 1
            break

    # Right vertical panel divider line: start scanning from near right edge inside the box rightwards
    scan_x_right = min(W - 1, max(cx, xmax - min(15, max(1, bw // 3))))
    limit_x2 = min(W - 1, xmax + pad_x)
    for x in range(scan_x_right, limit_x2 + 1):
        strip = gray[max(0, ymin - 5):min(H, ymax + 5), x]
        if len(strip) > 0 and np.mean(strip < 80) >= 0.70:
            limit_x2 = x - 1
            break

    return limit_x1, limit_y1, limit_x2, limit_y2


def refine_bubble_bbox(gray: "np.ndarray", xmin: int, ymin: int, xmax: int, ymax: int) -> Tuple[int, int, int, int]:
    """
    Expands OCR bounding box vertically to capture characters cut off by OCR truncation.
    Scans the vertical column above and below the box for character-sized components, stopping at panel divider lines.
    """
    if cv2 is None or np is None:
        return xmin, ymin, xmax, ymax
    H, W = gray.shape[:2]
    bw = xmax - xmin
    bh = ymax - ymin
    col_cx = (xmin + xmax) // 2

    lim_x1, lim_y1, lim_x2, lim_y2 = detect_panel_line_bounds(gray, xmin, ymin, xmax, ymax)
    pad_y = max(110, int(bh * 1.5))
    pad_x = max(30, int(bw * 0.40))
    sy1 = max(lim_y1, ymin - pad_y); sy2 = min(lim_y2, ymax + pad_y)
    sx1 = max(lim_x1, xmin - pad_x); sx2 = min(lim_x2, xmax + pad_x)
    if sy2 <= sy1 or sx2 <= sx1:
        return xmin, ymin, xmax, ymax

    roi = gray[sy1:sy2, sx1:sx2]
    dark = (roi < 150).astype(np.uint8) * 255
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    new_ymin, new_ymax, new_xmin, new_xmax = ymin, ymax, xmin, xmax
    for l in range(1, num_l):
        a = stats[l, cv2.CC_STAT_AREA]
        cw = stats[l, cv2.CC_STAT_WIDTH]
        ch = stats[l, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[l]
        abs_cx = cx + sx1
        abs_cy = cy + sy1
        abs_y1 = stats[l, cv2.CC_STAT_TOP] + sy1
        abs_y2 = abs_y1 + ch
        abs_x1 = stats[l, cv2.CC_STAT_LEFT] + sx1
        abs_x2 = abs_x1 + cw
        if abs(abs_cx - col_cx) <= max(bw * 0.70, 20):
            if 10 <= a <= 600 and cw <= max(bw * 0.85, 30) and ch <= max(bh * 0.50, 30):
                if sy1 <= abs_cy <= sy2:
                    new_ymin = max(lim_y1, min(new_ymin, abs_y1))
                    new_ymax = min(lim_y2, max(new_ymax, abs_y2))
                    new_xmin = max(lim_x1, min(new_xmin, abs_x1))
                    new_xmax = min(lim_x2, max(new_xmax, abs_x2))
    return new_xmin, new_ymin, new_xmax, new_ymax


def render_page_pipeline(
    page_img_path: str,
    output_img_path: str,
    trans_data: Dict[str, Any],
    target_lang_code: str = "en",
    keep_upscaled: bool = False,
) -> None:
    """Executes full 2-pass inpainting and typesetting for a single manga page."""
    img_bgr = cv2.imread(page_img_path, cv2.IMREAD_UNCHANGED) if cv2 is not None else None
    if img_bgr is None:
        shutil.copyfile(page_img_path, output_img_path)
        return

    orig_h, orig_w = img_bgr.shape[:2]
    scale_factor = 1.0
    if orig_h < 1400 and cv2 is not None:
        scale_factor = 1400.0 / orig_h
        new_w = int(round(orig_w * scale_factor))
        new_h = int(round(orig_h * scale_factor))
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Normalize image to 3-channel BGR
    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif len(img_bgr.shape) == 3:
        if img_bgr.shape[2] == 1:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 2:
            gray_ch = img_bgr[:, :, 0:1].astype(float)
            alpha = img_bgr[:, :, 1:2].astype(float) / 255.0
            white = np.ones_like(gray_ch) * 255.0
            gray_ch = (gray_ch * alpha + white * (1.0 - alpha)).astype(np.uint8)
            img_bgr = cv2.cvtColor(gray_ch, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            alpha = img_bgr[:, :, 3:4].astype(float) / 255.0
            bgr = img_bgr[:, :, :3].astype(float)
            white = np.ones_like(bgr) * 255.0
            img_bgr = (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)

    H, W = img_bgr.shape[:2]
    all_items = trans_data.get("items", [])
    raw_items = [it for it in all_items if not is_sfx_item(it)]

    bubble_items = []
    float_items = []

    for item in raw_items:
        c_type = item.get("container_type", "bubble")
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue
        if c_type in ("box", "bubble"):
            bubble_items.append(item)
        else:
            float_items.append(item)

    if not bubble_items and not float_items:
        Image.open(page_img_path).save(output_img_path)
        return

    # Pass 1a: Inpaint bubbles on img_bgr
    centroid_data = []
    for item in bubble_items:
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue
        y1_n, x1_n, y2_n, x2_n = box
        ymin = max(0, int(y1_n * H / 1000))
        xmin = max(0, int(x1_n * W / 1000))
        ymax = min(H, int(y2_n * H / 1000))
        xmax = min(W, int(x2_n * W / 1000))

        is_box = (item.get("container_type", "bubble") == "box")
        bg_hint = item.get("bg_color", "white")

        rx1, ry1, rx2, ry2 = xmin, ymin, xmax, ymax

        regions = inpaint_bubble(img_bgr, rx1, ry1, rx2, ry2, is_box=is_box, bg_color_hint=bg_hint)
        if not regions:
            float_items.append(item)
            continue

        # Conjoined lobe assignment:
        # If multiple lobes are detected for this cavity, assign the lobe closest to this item's text centroid
        if len(regions) > 1:
            item_cx = (xmin + xmax) // 2
            item_cy = (ymin + ymax) // 2
            closest_lobe = min(regions, key=lambda r: (r[0] - item_cx) ** 2 + (r[1] - item_cy) ** 2)
            regions = [closest_lobe]

        centroid_data.append((item, regions))

    # Pass 1a.2: Partition cavity envelopes of neighboring / conjoined bubbles to prevent text collision
    partition_overlapping_regions(centroid_data, H, W)

    # Pass 1b: Inpaint floating text on img_bgr
    floating_data = []
    for item in float_items:
        box = item.get("box_2d")
        if not box or len(box) != 4:
            continue
        y1_n, x1_n, y2_n, x2_n = box
        ymin = max(0, int(y1_n * H / 1000))
        xmin = max(0, int(x1_n * W / 1000))
        ymax = min(H, int(y2_n * H / 1000))
        xmax = min(W, int(x2_n * W / 1000))

        inpaint_floating(img_bgr, xmin, ymin, xmax, ymax)
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        bw = xmax - xmin
        bh = ymax - ymin
        max_allow_w = 2 * min(cx - 15, W - 15 - cx) if (cx > 15 and W - cx > 15) else W - 30

        if bh > bw * 2.0:
            sw = max(bw + 15, min(int(bw * 1.3), int(bh * 0.75), 140))
            sw = min(sw, max_allow_w)
            sh = bh + 20
            max_fs = min(22, max(12, int(bh * 0.10)))
        elif bh > bw * 1.2:
            sw = max(bw + 15, min(int(bh * 0.8), int(bw * 1.3), 150))
            sw = min(max(sw, 50), max(50, max_allow_w))
            sh = max(bh, 45)
            max_fs = min(24, max(13, int(bh * 0.16)))
        else:
            sw = max(bw, min(int(bh * 1.1), int(bw * 1.2), 150))
            sw = min(max(sw, 50), max(50, max_allow_w))
            sh = max(bh, int(bw * 0.8), 45)
            max_fs = min(26, max(14, int(bh * 0.25)))

        floating_data.append((item, cx, cy, sw, sh, max_fs))

    # Convert to PIL for precision typesetting
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Pass 2a: Typeset speech bubbles
    for item, regions in centroid_data:
        text = item.get("translated_text", "").strip()
        b_type = item.get("bubble_type", "dialogue")
        c_type = item.get("container_type", "bubble")
        if not text:
            continue
        font_path, sym_font_path = get_font_paths(target_lang_code, bubble_type=b_type, container_type=c_type)
        if len(regions) == 1:
            region = regions[0]
            cx, cy, sw, sh, text_color = region[0], region[1], region[2], region[3], region[4]
            profile = region[5] if len(region) > 5 else None
            bounds = region[6] if len(region) > 6 else None
            stroke_w = region[7] if len(region) > 7 else 0
            stroke_c = region[8] if len(region) > 8 else None
            draw_fitted(
                draw,
                text,
                cx,
                cy,
                sw,
                sh,
                font_path,
                sym_font_path,
                fill_color=text_color,
                target_lang_code=target_lang_code,
                bubble_type=b_type,
                container_type=c_type,
                profile=profile,
                bounds=bounds,
                stroke_width=stroke_w,
                stroke_color=stroke_c,
            )
        else:
            pairs = split_text_for_regions(text, regions)
            for chunk, region in pairs:
                cx, cy, sw, sh, text_color = region[0], region[1], region[2], region[3], region[4]
                profile = region[5] if len(region) > 5 else None
                bounds = region[6] if len(region) > 6 else None
                stroke_w = region[7] if len(region) > 7 else 0
                stroke_c = region[8] if len(region) > 8 else None
                if chunk.strip():
                    draw_fitted(
                        draw,
                        chunk,
                        cx,
                        cy,
                        sw,
                        sh,
                        font_path,
                        sym_font_path,
                        fill_color=text_color,
                        target_lang_code=target_lang_code,
                        bubble_type=b_type,
                        container_type=c_type,
                        profile=profile,
                        bounds=bounds,
                        stroke_width=stroke_w,
                        stroke_color=stroke_c,
                    )

    # Pass 2b: Typeset floating text
    for item, cx, cy, sw, sh, max_fs in floating_data:
        text = item.get("translated_text", "").strip()
        if not text:
            continue
        b_type = item.get("bubble_type", "dialogue")
        c_type = item.get("container_type", "floating")
        f_path, sym_path = get_font_paths(target_lang_code, bubble_type=b_type, container_type=c_type)
        draw_outlined(
            draw,
            text,
            cx,
            cy,
            sw,
            sh,
            f_path,
            sym_path,
            max_fs=max_fs,
            target_lang_code=target_lang_code,
            bubble_type=b_type,
            container_type=c_type,
        )

    # Save output image as high-quality WEBP
    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    if not keep_upscaled and scale_factor != 1.0:
        img_pil = img_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    img_pil.save(output_img_path, "WEBP", quality=95)


def load_resumed_pages(resume_path: str, temp_proc_dir: str) -> List[str]:
    """Extracts completed pages from an existing partial output file for resuming."""
    if not os.path.exists(resume_path):
        return []
    try:
        pages = extract_input_pages(resume_path, temp_proc_dir)
        return sorted(pages, key=natural_sort_key)
    except Exception as e:
        sys.stderr.write(f"    ⚠ Warning: Failed to read partial file for resume: {e}\n")
        return []


def process_manga(
    input_path: str,
    output_path: Optional[str] = None,
    target_lang_input: str = "English",
    source_lang_input: str = "Japanese",
    output_format: Optional[str] = None,
    key_manager: Optional[DynamicKeyManager] = None,
    export_script_path: Optional[str] = None,
    export_json_path: Optional[str] = None,
    skip_cover: bool = False,
    resume_path: Optional[str] = None,
    page_selection: Optional[str] = None,
    retrans_path: Optional[str] = None,
    keep_upscaled: bool = False,
) -> str:
    """
    Main entrypoint for translating a manga archive, image folder, or single page.
    Supports partial progress saving, automatic resume, and selective page re-translation.
    """
    start_time = time.time()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not key_manager:
        key_manager = DynamicKeyManager()

    tgt_name, tgt_code, _ = resolve_language_info(target_lang_input)
    src_name, src_code, _ = resolve_language_info(source_lang_input)

    filename = os.path.basename(input_path)
    base_name, in_ext = os.path.splitext(input_path)

    if not output_format:
        if in_ext in SUPPORTED_IMAGE_EXTS:
            format_ext = in_ext
        elif in_ext in SUPPORTED_ARCHIVE_EXTS or in_ext in SUPPORTED_DOC_EXTS:
            format_ext = in_ext
        else:
            format_ext = ".cbz"
    else:
        format_ext = f".{output_format.lstrip('.')}"

    if not output_path:
        out_file = f"{base_name}.{tgt_code}{format_ext}"
    else:
        out_file = output_path

    partial_file = get_partial_filepath(out_file)
    actual_resume_file = resume_path or (partial_file if os.path.exists(partial_file) else None)

    if resume_path and not os.path.exists(resume_path):
        sys.stderr.write(_yellow(f"  ⚠  --resume path not found: {resume_path}  (starting from scratch)\n"))
        actual_resume_file = None

    print_banner(f"Translating: {filename}")
    print(f"  {_bold('Source:')} {src_name} ({src_code})  →  {_bold('Target:')} {tgt_name} ({tgt_code})")
    print(f"  {_bold('Format:')} {format_ext.upper()}  |  {_bold('Output:')} {os.path.basename(out_file)}")
    if getattr(key_manager, "is_explicit_selection", False):
        k_count = len(key_manager.pool)
        labels = [k.get("label", mask_key(k.get("key", ""))) for k in key_manager.pool]
        k_str = ", ".join(labels[:4]) + (f" (+{k_count - 4} more)" if k_count > 4 else "")
        print(f"  {_bold('Keys:')}   {_cyan(str(k_count))} selected  ({_dim(k_str)})")
    if page_selection:
        retrans_note = f"  |  {_bold('Base:')} {os.path.basename(retrans_path)}" if retrans_path else ""
        print(f"  {_bold('Mode:')} {_cyan('Selective re-translation')}  ({_dim(page_selection)}){retrans_note}")
    elif actual_resume_file:
        print(f"  {_bold('Mode:')} {_cyan('Resume')}  ({os.path.basename(actual_resume_file)})")
    elif skip_cover:
        print(f"  {_bold('Cover:')} preserving original  {_dim('(--skip-cover)')}")
    print()

    work_dir = tempfile.mkdtemp(prefix="mangatrans_")
    raw_pages_dir = os.path.join(work_dir, "raw_pages")
    proc_pages_dir = os.path.join(work_dir, "processed_pages")
    retrans_dir = os.path.join(work_dir, "retrans_pages")
    os.makedirs(raw_pages_dir, exist_ok=True)
    os.makedirs(proc_pages_dir, exist_ok=True)

    narrative_mem = NarrativeMemory(max_pages=3)
    processed_page_paths: List[str] = []
    script_entries: List[str] = []
    all_json_entries: Dict[str, Any] = {}

    def _save_partial(label: str = "partial") -> None:
        if not processed_page_paths:
            print(
                _yellow(f"  To retry once quota resets:\n    {get_prog_name()} \"{input_path}\" -o \"{out_file}\"\n"),
                file=sys.stderr,
            )
            return
        try:
            pack_output_pages(processed_page_paths, partial_file, format_ext)
            print(_green(f"  ✔ Saved {label} progress ({len(processed_page_paths)}/{total_pages} pages) → {partial_file}"))
            if export_script_path:
                part_script = f"{os.path.splitext(export_script_path)[0]}.partial.md"
                with open(part_script, "w", encoding="utf-8") as sf:
                    sf.write(f"# Translated Script (Partial): {filename} ({tgt_name})\n\n" + "".join(script_entries))
                print(_green(f"  ✔ Saved partial script → {part_script}"))
            print(
                f"\n  {_bold('Resume command:')}\n    {get_prog_name()} \"{input_path}\" --resume \"{partial_file}\" -o \"{out_file}\"\n",
                file=sys.stderr,
            )
        except Exception as pack_err:
            sys.stderr.write(_yellow(f"  ⚠  Could not save partial file: {pack_err}\n"))

    def _print_key_status() -> None:
        cooldowns = load_key_cooldowns()
        now = time.time()
        print(_bold("\n  API Key Pool Status:"), file=sys.stderr)
        for i, k_info in enumerate(key_manager.pool):
            k_val = k_info.get("key", "")
            k_lbl = k_info.get("label", f"Key #{i+1}")
            cd_time = cooldowns.get(k_val, 0.0)
            if k_val in key_manager.permanently_failed:
                status_str = _red("✘ Invalid / Unauthorized")
            elif cd_time > now:
                status_str = _yellow(f"⏳ Cooldown (resets in {format_duration(cd_time - now)})")
            else:
                status_str = _green("✔ Ready")
            print(f"    {i+1:>2}. {k_lbl:<30} : {status_str}", file=sys.stderr)
        print(file=sys.stderr)

    try:
        sys.stdout.write(f"  {_bold('[1/4]')} Extracting manga pages... ")
        sys.stdout.flush()
        t0 = time.time()
        page_files = extract_input_pages(input_path, raw_pages_dir)
        total_pages = len(page_files)
        print(_green("done") + f" ({total_pages} page(s) in {time.time() - t0:.1f}s)")

        if total_pages == 0:
            raise RuntimeError("No image pages found in input.")

        if skip_cover and total_pages == 1:
            sys.stderr.write(_yellow("  ⚠  --skip-cover on a single-page input: output will be an untranslated copy.\n"))

        if page_selection:
            selected_page_nums: Set[int] = set(parse_page_selection(page_selection, total_pages))
            if not selected_page_nums:
                raise RuntimeError("--pages spec resolved to no valid pages.")
            print(
                f"  {_cyan('Selective mode:')} re-translating {len(selected_page_nums)} page(s): "
                + _dim(", ".join(str(p) for p in sorted(selected_page_nums)))
            )
        else:
            selected_page_nums = set(range(1, total_pages + 1))

        existing_translated: Dict[int, str] = {}
        if retrans_path and os.path.exists(retrans_path):
            os.makedirs(retrans_dir, exist_ok=True)
            try:
                retrans_pages = extract_input_pages(retrans_path, retrans_dir)
                for i, rp in enumerate(retrans_pages):
                    existing_translated[i + 1] = rp
                print(
                    f"  {_green('✔')} Loaded {len(existing_translated)} existing page(s) from: "
                    + _dim(os.path.basename(retrans_path))
                )
            except Exception as re_err:
                sys.stderr.write(
                    _yellow(f"  ⚠  Could not read --retrans file: {re_err}  (will copy originals for non-selected pages)\n")
                )
        elif retrans_path:
            sys.stderr.write(
                _yellow(f"  ⚠  --retrans path not found: {retrans_path}  (will copy originals for non-selected pages)\n")
            )

        resumed_count = 0
        if actual_resume_file and os.path.isfile(actual_resume_file) and not page_selection:
            resumed_pages = load_resumed_pages(actual_resume_file, proc_pages_dir)
            resumed_count = min(len(resumed_pages), total_pages)
            if resumed_count > 0:
                print(f"  {_green('✔')} Resuming: {resumed_count}/{total_pages} page(s) already done.")
                processed_page_paths = resumed_pages[:resumed_count]

        print(f"\n  {_bold('[2/4]')} Translating & Typesetting {total_pages} page(s)...")

        def _fetch_page(p_path: str, context_prompt: str, cur_progress: Optional[LiveProgress] = None) -> Dict[str, Any]:
            with open(p_path, "rb") as pf:
                b = pf.read()
            return query_page_translation(
                image_bytes=b,
                key_manager=key_manager,
                src_lang_name=src_name,
                tgt_lang_name=tgt_name,
                src_lang_code=src_code,
                narrative_context=context_prompt,
                progress=cur_progress,
            )

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        next_future = None
        next_future_idx = None

        try:
            digits = len(str(total_pages))
            for idx in range(resumed_count, total_pages):
                page_p = page_files[idx]
                p_num = idx + 1
                p_name = os.path.basename(page_p)
                out_page_path = os.path.join(proc_pages_dir, f"{p_num:04d}.webp")

                if p_num not in selected_page_nums:
                    src_for_copy = existing_translated.get(p_num, page_p)
                    shutil.copyfile(src_for_copy, out_page_path)
                    processed_page_paths.append(out_page_path)
                    continue

                if skip_cover and p_num == 1:
                    progress = LiveProgress(f"[{p_num:>{digits}}/{total_pages}] Preserving cover ({p_name})...")
                    progress.start()
                    orig_img = Image.open(page_p)
                    orig_img.save(out_page_path, "WEBP", quality=95)
                    processed_page_paths.append(out_page_path)
                    progress.stop("preserved")
                    for nxt_idx in range(idx + 1, total_pages):
                        nxt_p_num = nxt_idx + 1
                        if nxt_p_num in selected_page_nums and not (skip_cover and nxt_p_num == 1):
                            next_future = executor.submit(
                                _fetch_page, page_files[nxt_idx], narrative_mem.get_context_prompt()
                            )
                            next_future_idx = nxt_idx
                            break
                    continue

                key_disp = key_manager.get_key_status_str()
                base_label = f"[{p_num:>{digits}}/{total_pages}] {key_disp}  Processing {p_name}..."
                progress = LiveProgress(base_label, base_label=f"[{p_num:>{digits}}/{total_pages}] Processing {p_name}...")
                progress.start()

                try:
                    if next_future is not None and next_future_idx == idx:
                        trans_data = next_future.result()
                    else:
                        trans_data = _fetch_page(page_p, narrative_mem.get_context_prompt(), cur_progress=progress)
                    items_count = len(trans_data.get("items", []))
                    narrative_mem.add_page(p_num, trans_data.get("items", []))

                except AllKeysExhaustedError:
                    progress.stop("all keys exhausted")
                    print(_red("\n  ✘ ALL API keys have hit their rate/quota limits."), file=sys.stderr)
                    _print_key_status()
                    _save_partial()
                    sys.exit(2)

                except Exception as e:
                    progress.set_message(f"[{p_num:>{digits}}/{total_pages}] Retrying {p_name} ({e})...")
                    try:
                        trans_data = _fetch_page(page_p, narrative_mem.get_context_prompt(), cur_progress=progress)
                        items_count = len(trans_data.get("items", []))
                        narrative_mem.add_page(p_num, trans_data.get("items", []))
                    except AllKeysExhaustedError:
                        progress.stop("all keys exhausted")
                        print(_red("\n  ✘ ALL API keys exhausted during retry."), file=sys.stderr)
                        _print_key_status()
                        _save_partial()
                        sys.exit(2)
                    except Exception as e2:
                        progress.set_message(f"[{p_num:>{digits}}/{total_pages}] Query error ({e2})...")
                        sys.stderr.write(_yellow(f"    ⚠  Page #{p_num}: query error ({e2}) — copying original.\n"))
                        trans_data = {"items": []}
                        items_count = 0
                finally:
                    next_future = None
                    next_future_idx = None

                for nxt_idx in range(idx + 1, total_pages):
                    nxt_p_num = nxt_idx + 1
                    if nxt_p_num in selected_page_nums and not (skip_cover and nxt_p_num == 1):
                        nxt_p = page_files[nxt_idx]
                        next_future = executor.submit(_fetch_page, nxt_p, narrative_mem.get_context_prompt())
                        next_future_idx = nxt_idx
                        progress.set_message(
                            f"[{p_num:>{digits}}/{total_pages}] Typesetting {p_name} "
                            f"({items_count} item(s)) | Prefetching {os.path.basename(nxt_p)}..."
                        )
                        break
                else:
                    progress.set_message(
                        f"[{p_num:>{digits}}/{total_pages}] Typesetting & Inpainting {p_name} ({items_count} item(s))..."
                    )

                render_page_pipeline(page_p, out_page_path, trans_data, target_lang_code=tgt_code, keep_upscaled=keep_upscaled)
                processed_page_paths.append(out_page_path)
                model_used = trans_data.get("_model_used", "")
                model_tag = f" via {_cyan(model_used)}" if model_used else ""
                progress.stop(f"done{model_tag} ({items_count} item(s))")

                all_json_entries[p_name] = trans_data
                if export_json_path:
                    import json
                    if os.path.isdir(export_json_path) or not export_json_path.lower().endswith(".json"):
                        os.makedirs(export_json_path, exist_ok=True)
                        p_stem, p_ext = os.path.splitext(p_name)
                        cand_name = f"{p_stem}.json"
                        cand_path = os.path.join(export_json_path, cand_name)
                        # Avoid collision if multiple extensions share the same stem
                        if os.path.exists(cand_path):
                            cand_name = f"{p_name}.json"
                            cand_path = os.path.join(export_json_path, cand_name)
                        with open(cand_path, "w", encoding="utf-8") as jf:
                            json.dump(trans_data, jf, indent=2, ensure_ascii=False)
                    elif total_pages == 1:
                        os.makedirs(os.path.dirname(os.path.abspath(export_json_path)) or ".", exist_ok=True)
                        with open(export_json_path, "w", encoding="utf-8") as jf:
                            json.dump(trans_data, jf, indent=2, ensure_ascii=False)

                items = trans_data.get("items", [])
                if items:
                    script_entries.append(f"### Page {p_num}\n")
                    for it in items:
                        b_type = it.get("bubble_type", "dialogue").upper()
                        orig_t = it.get("original_text", "").strip()
                        trans_t = it.get("translated_text", "").strip()
                        script_entries.append(f"- **[{b_type}]** {trans_t} *(Source: {orig_t})*\n")
                    script_entries.append("\n")

        except KeyboardInterrupt:
            sys.stderr.write("\n")
            _save_partial("interrupted")
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if export_json_path and export_json_path.lower().endswith(".json") and total_pages > 1:
            import json
            os.makedirs(os.path.dirname(os.path.abspath(export_json_path)) or ".", exist_ok=True)
            with open(export_json_path, "w", encoding="utf-8") as jf:
                json.dump(all_json_entries, jf, indent=2, ensure_ascii=False)
            print(f"  {_bold('✔')} Combined JSON exported: {_dim(export_json_path)}")

        if export_script_path:
            with open(export_script_path, "w", encoding="utf-8") as sf:
                sf.write(f"# Translated Script: {filename} ({tgt_name})\n\n" + "".join(script_entries))
            print(f"\n  {_bold('[3/4]')} Script exported: {_dim(export_script_path)}")
        else:
            print(f"\n  {_bold('[3/4]')} Typesetting complete for all {total_pages} page(s).")

        sys.stdout.write(f"  {_bold('[4/4]')} Repackaging into {format_ext.upper()} container... ")
        sys.stdout.flush()
        final_output = pack_output_pages(processed_page_paths, out_file, format_ext)
        print(_green("done") + f" → {_bold(final_output)}")

        if os.path.isfile(partial_file):
            try:
                os.remove(partial_file)
            except Exception:
                pass

        total_time = time.time() - start_time
        mins, secs = divmod(int(total_time), 60)
        time_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
        avg_per_page = total_time / max(1, total_pages)
        print(
            f"\n  {_green('✔')} {_bold('Finished:')} {total_pages} page(s) in {_bold(time_str)} ({avg_per_page:.1f}s/page)"
            f"  →  {_bold(os.path.basename(final_output))}\n"
        )

        return final_output

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def convert_manga_container(input_path: str, output_format: str, output_path: Optional[str] = None) -> str:
    """Converts a manga container/folder between formats without translating."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    base, _ = os.path.splitext(input_path)
    target_ext = f".{output_format.lstrip('.')}"
    if not output_path:
        output_path = f"{base}{target_ext}"

    print(f"◆ Converting '{input_path}' → '{output_path}' ({target_ext.upper()})...")
    work_dir = tempfile.mkdtemp(prefix="manga_conv_")
    try:
        pages = extract_input_pages(input_path, work_dir)
        if not pages:
            raise RuntimeError("No image pages found to convert.")
        res = pack_output_pages(pages, output_path, target_ext)
        print(f"[✔] Converted {len(pages)} page(s) successfully -> {res}")
        return res
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
