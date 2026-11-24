"""Syllable hyphenation and scanlation word-wrapping engine."""

import re
from typing import Dict, List, Tuple

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    ImageDraw = None
    ImageFont = None


# Common English prefix and suffix split patterns for natural typography
COMMON_PREFIXES = ("trans", "inter", "micro", "super", "under", "over", "auto", "anti", "pre", "pro", "dis", "mis", "un", "re", "in", "ex", "de")
COMMON_SUFFIXES = ("tional", "ssion", "ation", "ment", "ness", "able", "ible", "fully", "less", "ting", "ling", "ring", "ping", "sing", "ding", "king", "ing", "tion", "sion", "ous", "ies", "ted", "ded", "ly")


def find_hyphenation_points(word: str) -> List[int]:
    """
    Finds typographical hyphenation split indices for an English word.
    Returns list of 1-based character indices where hyphenation is visually natural.
    """
    w = word.strip()
    core_w = w.rstrip("-!?,.:;~♪")
    if len(core_w) <= 4:
        return []

    # If word already contains an internal hyphen, split there
    if "-" in core_w:
        return [i + 1 for i, c in enumerate(core_w) if c == "-"]

    lower_w = core_w.lower()
    points = []

    # 1. Check prefixes
    for pref in COMMON_PREFIXES:
        if lower_w.startswith(pref) and len(lower_w) - len(pref) >= 3:
            points.append(len(pref))
            break

    # 2. Check suffixes
    for suff in COMMON_SUFFIXES:
        if lower_w.endswith(suff) and len(lower_w) - len(suff) >= 3:
            split_idx = len(lower_w) - len(suff)
            if split_idx not in points:
                points.append(split_idx)
            break

    # 3. Double consonants (e.g. but-ter, hap-py, mes-sage)
    vowels = set("aeiouy")
    for i in range(2, len(core_w) - 2):
        c1, c2 = lower_w[i - 1], lower_w[i]
        if c1 == c2 and c1 not in vowels and c1 not in ("l", "s"):
            if i not in points:
                points.append(i)

    # 4. Vowel-Consonant-Consonant-Vowel (VC-CV) pattern
    for i in range(2, len(core_w) - 2):
        if (
            lower_w[i - 2] in vowels
            and lower_w[i - 1] not in vowels
            and lower_w[i] not in vowels
            and lower_w[i + 1] in vowels
        ):
            if i not in points:
                points.append(i)

    # Filter points too close to ends
    valid = [p for p in points if 2 <= p <= len(core_w) - 2]
    if not valid and len(core_w) >= 7:
        # Fallback midpoint split
        mid = len(core_w) // 2
        valid = [mid]

    valid.sort()
    return valid


_TEXT_LEN_CACHE: Dict[Tuple[str, int, int], float] = {}


def measure_line_length(
    draw: "ImageDraw.ImageDraw",
    line: str,
    font: "ImageFont.ImageFont",
    sym_font: "ImageFont.ImageFont",
) -> float:
    """Calculates exact typographic advance width of a line of text with memoization."""
    key = (line, id(font), id(sym_font))
    cached = _TEXT_LEN_CACHE.get(key)
    if cached is not None:
        return cached

    syms = set("♥♡♫♪★☆~")
    if not any(c in syms for c in line):
        total_w = float(draw.textlength(line, font=font))
    else:
        total_w = 0.0
        cur_run = []
        cur_is_sym = False
        for ch in line:
            is_sym = (ch in syms)
            if is_sym != cur_is_sym and cur_run:
                chunk = "".join(cur_run)
                f = sym_font if cur_is_sym else font
                total_w += float(draw.textlength(chunk, font=f))
                cur_run = []
            cur_is_sym = is_sym
            cur_run.append(ch)
        if cur_run:
            chunk = "".join(cur_run)
            f = sym_font if cur_is_sym else font
            total_w += float(draw.textlength(chunk, font=f))

    if len(_TEXT_LEN_CACHE) > 50000:
        _TEXT_LEN_CACHE.clear()
    _TEXT_LEN_CACHE[key] = total_w
    return total_w


def wrap_words_for_width(
    draw: "ImageDraw.ImageDraw",
    words: List[str],
    font: "ImageFont.ImageFont",
    sym_font: "ImageFont.ImageFont",
    avail_w: int,
    allow_hyphenation: bool = True,
) -> Tuple[List[str], bool]:
    """
    Wraps words into lines fitting within avail_w.
    Uses syllable hyphenation when a long word overflows to maintain larger, readable font sizes.
    """
    lines = []
    cur = []
    bad_overflow = False

    for w in words:
        test = " ".join(cur + [w]) if cur else w
        w_len = measure_line_length(draw, test, font, sym_font)
        if w_len <= avail_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = []

            # Now test if single word w fits on a fresh line
            single_len = measure_line_length(draw, w, font, sym_font)
            if single_len <= avail_w:
                cur = [w]
            else:
                # Word itself exceeds line width: attempt hyphenation split
                splited = False
                if allow_hyphenation and avail_w > 18 and len(w) >= 5:
                    split_points = find_hyphenation_points(w)
                    # Try split points from right to left to maximize first line content
                    for pt in reversed(split_points):
                        p1 = w[:pt] + ("-" if not w[:pt].endswith("-") else "")
                        p2 = w[pt:]
                        if measure_line_length(draw, p1, font, sym_font) <= avail_w:
                            lines.append(p1)
                            if measure_line_length(draw, p2, font, sym_font) <= avail_w:
                                cur = [p2]
                                splited = True
                                break
                            else:
                                sub_pts = find_hyphenation_points(p2)
                                sub_split = False
                                for s_pt in reversed(sub_pts):
                                    sp1 = p2[:s_pt] + ("-" if not p2[:s_pt].endswith("-") else "")
                                    sp2 = p2[s_pt:]
                                    if measure_line_length(draw, sp1, font, sym_font) <= avail_w and measure_line_length(draw, sp2, font, sym_font) <= avail_w:
                                        lines.append(sp1)
                                        cur = [sp2]
                                        sub_split = True
                                        break
                                if sub_split:
                                    splited = True
                                    break
                                else:
                                    lines.append(p2)
                                    bad_overflow = True
                                    splited = True
                                    break

                if not splited:
                    bad_overflow = True
                    lines.append(w)

    if cur:
        lines.append(" ".join(cur))

    return lines, bad_overflow
