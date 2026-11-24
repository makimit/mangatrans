"""Scanlation typography, font routing, symbol rendering, and halo outlining."""

import os
from typing import List, Optional, Tuple

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    ImageDraw = None
    ImageFont = None

from ..config import FONTS_DIR, ensure_scanlation_fonts, find_system_fallback_font
from .hyphenation import measure_line_length, wrap_words_for_width


def get_font_paths(
    lang_code: str = "en",
    bubble_type: str = "dialogue",
    container_type: str = "bubble",
) -> Tuple[str, str]:
    """Returns (primary_font_path, symbol_font_path) tailored to target language script and bubble semantic role."""
    ensure_scanlation_fonts()
    primary_font = os.path.join(FONTS_DIR, "AnimeAce2_Bold.ttf")

    if lang_code in ("hi", "hin", "mr", "ne", "sa"):
        dev_font = os.path.join(FONTS_DIR, "NotoSansDevanagari.ttf")
        if os.path.isfile(dev_font):
            primary_font = dev_font
    elif lang_code in ("bn", "ben", "as"):
        ben_font = os.path.join(FONTS_DIR, "NotoSansBengali.ttf")
        if os.path.isfile(ben_font):
            primary_font = ben_font
    elif lang_code in ("ru", "rus", "uk", "ukr", "el", "grc", "zh", "ko", "ja"):
        noto_font = os.path.join(FONTS_DIR, "NotoSans.ttf")
        if os.path.isfile(noto_font):
            primary_font = noto_font
    else:
        # Latin / English semantic typography hierarchy
        b_type = (bubble_type or "dialogue").lower()
        c_type = (container_type or "bubble").lower()

        if b_type in ("whisper", "aside"):
            # Lighter italic comic style for whisper / aside
            for candidate in [
                os.path.join(FONTS_DIR, "ComicNeue-Italic.ttf"),
                os.path.join(FONTS_DIR, "ComicNeue-Regular.ttf"),
                os.path.join(FONTS_DIR, "ComicNeueAngular-Italic.ttf"),
            ]:
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 1000:
                    primary_font = candidate
                    break
        else:
            # Use uniform scanlation font (AnimeAce2_Bold) for dialogue, thoughts, boxes, and rectangles
            primary_font = os.path.join(FONTS_DIR, "AnimeAce2_Bold.ttf")

    if not os.path.isfile(primary_font) or os.path.getsize(primary_font) < 1000:
        fallback = find_system_fallback_font(is_symbol=False)
        if fallback:
            primary_font = fallback

    sym_font = find_system_fallback_font(is_symbol=True)
    if not sym_font:
        sym_font = primary_font

    return primary_font, sym_font


def normalize_scanlation_text(text: str) -> str:
    """Normalizes dashes, quotes, and punctuation symbols lacking in classic comic fonts."""
    if not text:
        return ""
    text = text.replace("——", "--").replace("――", "--").replace("—", "-").replace("―", "-")
    text = text.replace("–", "-").replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    text = text.replace("…", "...").replace("〜", "~").replace("～", "~")
    # Strip artificial leading hesitation dashes if the text is a single short exclamation
    if text.startswith(("-", "--")) and len(text.split()) <= 2:
        text = text.lstrip("-").strip()
    return text


def format_text_casing(
    text: str,
    target_lang_code: str = "en",
    bubble_type: str = "dialogue",
    container_type: str = "bubble",
) -> str:
    """Formats text capitalization smartly based on target language and dialogue type."""
    if not text:
        return ""
    if target_lang_code in ("en", "eng"):
        b_type = (bubble_type or "dialogue").lower()
        c_type = (container_type or "bubble").lower()
        if c_type == "box" or b_type in ("whisper", "aside", "narration", "box"):
            return text
        return text.upper()
    return text


def draw_line_with_symbols(
    draw: "ImageDraw.ImageDraw",
    line: str,
    x: int,
    y: int,
    fill: Tuple[int, int, int],
    font: "ImageFont.ImageFont",
    sym_font: "ImageFont.ImageFont",
    stroke_width: int = 0,
    stroke_fill: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draws a single line of text with native TrueType kerning, glyph routing for symbols, and optional halo glow stroke."""
    syms = set("♥♡♫♪★☆~")
    if not any(c in syms for c in line):
        draw.text(
            (x, y),
            line,
            fill=fill,
            font=font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return

    curr_x = x
    cur_run = []
    cur_is_sym = False

    for ch in line:
        is_sym = (ch in syms)
        if is_sym != cur_is_sym and cur_run:
            chunk = "".join(cur_run)
            f = sym_font if cur_is_sym else font
            draw.text(
                (curr_x, y),
                chunk,
                fill=fill,
                font=f,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            curr_x += int(draw.textlength(chunk, font=f))
            cur_run = []
        cur_is_sym = is_sym
        cur_run.append(ch)

    if cur_run:
        chunk = "".join(cur_run)
        f = sym_font if cur_is_sym else font
        draw.text(
            (curr_x, y),
            chunk,
            fill=fill,
            font=f,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )


def draw_outlined(
    draw: "ImageDraw.ImageDraw",
    text: str,
    cx: int,
    cy: int,
    sw: int,
    sh: int,
    font_path: str,
    sym_font_path: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    outline_color: Tuple[int, int, int] = (255, 255, 255),
    max_fs: int = 48,
    min_fs: int = 11,
    target_lang_code: str = "en",
    bubble_type: str = "dialogue",
    container_type: str = "floating",
) -> None:
    """
    Renders floating text with an 8-direction halo outline for maximum legibility over artwork.
    Decouples tall, narrow vertical Japanese text boxes into readable horizontal aspect ratios.
    """
    formatted = format_text_casing(
        normalize_scanlation_text(text),
        target_lang_code=target_lang_code,
        bubble_type=bubble_type,
        container_type=container_type,
    )
    words = formatted.split()
    if not words:
        return

    canvas_w = draw.im.size[0] if hasattr(draw, "im") and draw.im else 1280
    canvas_h = draw.im.size[1] if hasattr(draw, "im") and draw.im else 1780

    # Aspect ratio decoupling: vertical Japanese text needs natural horizontal width in English.
    # Allow expanding up to 210px or sh * 0.65; start_x clamping below ensures it never clips canvas boundaries.
    if sh > sw * 1.8:
        target_w = max(sw, min(max(int(sw * 2.2), int(sh * 0.65), 140), 210, canvas_w - 40))
        sw = target_w
        sh = max(40, int(sh * 0.85))
    else:
        sw = min(sw, canvas_w - 40)

    best = None
    computed_max = min(max_fs, 28, max(min_fs, int(max(sw, sh) * 0.25)))

    for fs in range(computed_max, min_fs - 1, -1):
        try:
            f = ImageFont.truetype(font_path, max(4, min(72, fs)))
        except Exception:
            f = ImageFont.load_default()
        try:
            f_sym = ImageFont.truetype(sym_font_path, max(4, min(84, int(fs * 1.15))))
        except Exception:
            f_sym = f

        lines, bad_overflow = wrap_words_for_width(draw, words, f, f_sym, sw, allow_hyphenation=True)
        if bad_overflow:
            continue

        sample_bb = draw.textbbox((0, 0), "Ag", font=f)
        lh = int((sample_bb[3] - sample_bb[1]) * 1.20)
        gap = max(2, int(fs * 0.20))
        total_h = len(lines) * lh + max(0, len(lines) - 1) * gap

        if total_h <= sh:
            best = (f, f_sym, fs, lines, lh, gap)
            break

    if best is None:
        fs = max(4, min(72, min_fs))
        try:
            f = ImageFont.truetype(font_path, fs)
        except Exception:
            f = ImageFont.load_default()
        try:
            f_sym = ImageFont.truetype(sym_font_path, max(4, min(84, int(fs * 1.15))))
        except Exception:
            f_sym = f
        lines, _ = wrap_words_for_width(draw, words, f, f_sym, sw, allow_hyphenation=True)
        sample_bb = draw.textbbox((0, 0), "Ag", font=f)
        lh = int((sample_bb[3] - sample_bb[1]) * 1.20)
        gap = max(2, int(fs * 0.20))
        best = (f, f_sym, fs, lines, lh, gap)

    f_main, f_sym, fs, lines, lh, gap = best
    total_h = len(lines) * lh + max(0, len(lines) - 1) * gap
    outline_thickness = max(2, int(fs * 0.12))
    offsets = [
        (dx, dy)
        for dx in range(-outline_thickness, outline_thickness + 1)
        for dy in range(-outline_thickness, outline_thickness + 1)
        if dx != 0 or dy != 0
    ]

    # Pass 1: Halo outline
    curr_y = cy - total_h // 2
    for line in lines:
        line_w = measure_line_length(draw, line, f_main, f_sym)
        start_x = int(cx - line_w / 2)
        if start_x + line_w > canvas_w - 10:
            start_x = max(10, canvas_w - 10 - int(line_w))
        if start_x < 10:
            start_x = 10
        for dx, dy in offsets:
            draw_line_with_symbols(draw, line, start_x + dx, curr_y + dy, outline_color, f_main, f_sym)
        curr_y += lh + gap

    # Pass 2: Foreground crisp text
    curr_y = cy - total_h // 2
    for line in lines:
        line_w = measure_line_length(draw, line, f_main, f_sym)
        start_x = int(cx - line_w / 2)
        if start_x + line_w > canvas_w - 10:
            start_x = max(10, canvas_w - 10 - int(line_w))
        if start_x < 10:
            start_x = 10
        draw_line_with_symbols(draw, line, start_x, curr_y, text_color, f_main, f_sym)
        curr_y += lh + gap
