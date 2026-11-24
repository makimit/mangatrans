"""Cavity layout fitting, volume filling, and conjoined text distribution."""

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    ImageDraw = None
    ImageFont = None

try:
    import numpy as np
except ImportError:
    np = None

from .hyphenation import measure_line_length, wrap_words_for_width
from .typography import draw_line_with_symbols, format_text_casing, normalize_scanlation_text


def dp_balance_lines(
    draw: "ImageDraw.ImageDraw",
    words: List[str],
    f: Any,
    f_sym: Any,
    k: int,
    line_target_widths: List[int],
) -> Optional[List[str]]:
    """
    Dynamic Programming Knuth-Plass line balancing:
    Partitions N words into exactly k lines to match target line widths (e.g. oval bubble profile)
    minimizing variance and eliminating dangling orphan words.
    """
    N = len(words)
    if k <= 0 or N == 0:
        return None
    if k == 1:
        line_str = " ".join(words)
        lw = measure_line_length(draw, line_str, f, f_sym)
        if lw <= line_target_widths[0]:
            return [line_str]
        return None
    if k > N:
        return None

    word_lens = [measure_line_length(draw, w, f, f_sym) for w in words]
    space_w = measure_line_length(draw, " ", f, f_sym)

    memo: Dict[Tuple[int, int], Tuple[float, int]] = {}

    def cost(line_idx: int, word_start: int) -> Tuple[float, int]:
        if line_idx == k:
            return (0.0, word_start) if word_start == N else (float("inf"), -1)
        if word_start == N:
            return (float("inf"), -1)
        state = (line_idx, word_start)
        if state in memo:
            return memo[state]

        rem_lines = k - line_idx
        rem_words = N - word_start
        if rem_words < rem_lines:
            return (float("inf"), -1)

        target_w = line_target_widths[line_idx]
        best_c = float("inf")
        best_next = -1

        curr_w = 0.0
        for end in range(word_start + 1, N - rem_lines + 2):
            if end == word_start + 1:
                curr_w = word_lens[word_start]
            else:
                curr_w += space_w + word_lens[end - 1]

            if curr_w > target_w * 1.05 and end > word_start + 1:
                break

            slack = target_w - curr_w
            slack_pen = (slack ** 2) if slack >= 0 else (slack ** 2 * 20.0)

            orphan_pen = 0.0
            line_word_count = end - word_start
            if line_idx == k - 1 and line_word_count == 1 and len(words[word_start]) <= 4:
                orphan_pen = 50000.0
            elif line_word_count == 1 and N > 3:
                orphan_pen = 20000.0

            sub_c, _ = cost(line_idx + 1, end)
            if sub_c < float("inf"):
                total_c = slack_pen + orphan_pen + sub_c
                if total_c < best_c:
                    best_c = total_c
                    best_next = end

        memo[state] = (best_c, best_next)
        return memo[state]

    tot_cost, first_split = cost(0, 0)
    if tot_cost == float("inf"):
        return None

    res = []
    curr = 0
    for l_idx in range(k):
        _, next_curr = memo.get((l_idx, curr), (0, N))
        if next_curr == -1 or next_curr <= curr:
            next_curr = min(N, curr + 1)
        res.append(" ".join(words[curr:next_curr]))
        curr = next_curr
    return res


def _measure_or_draw(
    draw: "ImageDraw.ImageDraw",
    text: str,
    cx: int,
    cy: int,
    avail_w: int,
    avail_h: int,
    font_path: str,
    sym_font_path: str,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
    draw_it: bool = False,
    max_fs: int = 56,
    min_fs: int = 4,
    target_lang_code: str = "en",
    bubble_type: str = "dialogue",
    container_type: str = "bubble",
    profile: Optional[Dict[int, Tuple[int, int]]] = None,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    stroke_width: int = 0,
    stroke_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bool, int]:
    """
    Core scanlation typesetting engine:
    - Iterates font size down from max_fs with dynamic line partitioning to maximize bubble space utilization.
    - Syllabic hyphenation prevents premature font shrinking.
    - Centers lines dynamically within the cavity profile.
    - Strictly checks every text line against the actual 360-degree cavity boundary profile.
    - Prevents any line from clipping or touching the bubble border.
    """
    formatted = format_text_casing(
        normalize_scanlation_text(text),
        target_lang_code=target_lang_code,
        bubble_type=bubble_type,
    )
    raw_words = formatted.split()
    if not raw_words:
        return True, 0

    # Expand hyphenated compound words so typesetting can wrap after hyphens
    words = []
    for w in raw_words:
        if "-" in w and len(w) > 3 and not w.startswith("-") and not w.endswith("-"):
            parts = w.split("-")
            for i, p in enumerate(parts):
                if i < len(parts) - 1:
                    words.append(p + "-")
                else:
                    if p:
                        words.append(p)
        else:
            words.append(w)

    min_c, min_r, max_c, max_r = (
        bounds
        if bounds
        else (
            cx - avail_w // 2,
            cy - avail_h // 2,
            cx + avail_w // 2,
            cy + avail_h // 2,
        )
    )
    is_box = (bubble_type in ("narration", "box")) or (container_type == "box")
    bubble_w = max_c - min_c
    bubble_h = max_r - min_r
    N = len(words)
    is_vertical = (bubble_h > 1.30 * bubble_w)
    min_wpl = 0.95 if is_vertical else 1.30

    best = None
    best_score = -1e9
    computed_max = min(max_fs, 72, max(min_fs, int(max(avail_w, min(avail_w, avail_h)) * 0.85)))

    ratios = [1.35, 1.25, 1.15, 1.08, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.55]

    for fs in range(computed_max, min_fs - 1, -1):
        try:
            f = ImageFont.truetype(font_path, max(4, min(72, fs)))
        except Exception:
            f = ImageFont.load_default()
        try:
            f_sym = ImageFont.truetype(sym_font_path, max(4, min(84, int(fs * 1.15))))
        except Exception:
            f_sym = f

        sample_bb = draw.textbbox((0, 0), "Ag", font=f)
        lh = int((sample_bb[3] - sample_bb[1]) * 1.15)
        line_gap = max(1, int(fs * 0.18))
        M = max(7, min(14, int(fs * 0.35))) if not is_box else max(2, int(fs * 0.12))
        v_air = max(4, int(bubble_h * 0.08)) if (not is_box and bubble_h > 50) else 0

        candidates = []

        # 1. Knuth-Plass DP Line Balancing Candidates
        max_k = max(1, (bubble_h - 2 * M - 2 * v_air + line_gap) // max(1, lh + line_gap))
        for k_cand in range(1, min(N + 1, max_k + 1)):
            if is_box:
                target_widths = [avail_w] * k_cand
            else:
                target_widths = [
                    max(20, int(avail_w * (1.0 - 0.22 * ((2.0 * i - (k_cand - 1)) / max(1.0, k_cand - 0.5)) ** 2)))
                    for i in range(k_cand)
                ]
            dp_cand = dp_balance_lines(draw, words, f, f_sym, k_cand, target_widths)
            if dp_cand and dp_cand not in candidates:
                candidates.append(dp_cand)

        # 2. Aspect-ratio greedy candidates
        for r in ratios:
            tw = max(20, int(avail_w * r))
            lines, bad = wrap_words_for_width(draw, words, f, f_sym, tw, allow_hyphenation=False)
            if not bad and lines not in candidates:
                candidates.append(lines)

        # 3. LastResort hyphenation (only when unhyphenated candidates don't exist and fs < 14)
        if not candidates or fs < 12:
            for r in ratios:
                tw = max(20, int(avail_w * r))
                lines, bad = wrap_words_for_width(draw, words, f, f_sym, tw, allow_hyphenation=True)
                if not bad and lines not in candidates:
                    candidates.append(lines)

        if N == 1 and bubble_h > 1.7 * bubble_w and len(words[0]) >= 3:
            cand_v = list(words[0])
            if cand_v not in candidates:
                candidates.append(cand_v)
        elif N == 2:
            cand2 = [words[0], words[1]]
            if cand2 not in candidates:
                candidates.append(cand2)
        elif N == 3:
            for cand3 in [
                [words[0], f"{words[1]} {words[2]}"],
                [f"{words[0]} {words[1]}", words[2]],
                [words[0], words[1], words[2]],
            ]:
                if cand3 not in candidates:
                    candidates.append(cand3)

        for raw_lines in candidates:
            lines = [" ".join(p) if isinstance(p, list) else str(p) for p in raw_lines]
            k = len(lines)
            tot_h = k * lh + (k - 1) * line_gap
            if tot_h > (bubble_h - 2 * M):
                continue

            y_top_cands = [cy - tot_h // 2]
            if is_vertical and (bubble_h - tot_h) > (2 * M + 8):
                top_aligned = min_r + max(M + 2, int(lh * 0.40))
                mid_top = (min_r + M + cy - tot_h // 2) // 2
                if top_aligned not in y_top_cands:
                    y_top_cands.append(top_aligned)
                if mid_top not in y_top_cands:
                    y_top_cands.append(mid_top)

            for y_top in y_top_cands:
                y_bot = y_top + tot_h
                if y_top < min_r + M or y_bot > max_r - M:
                    continue

                fits = True
                min_clearance = 999.0
                for idx, line in enumerate(lines):
                    lw = measure_line_length(draw, line, f, f_sym)
                    line_y1 = y_top + idx * (lh + line_gap)
                    line_y2 = line_y1 + lh
                    if profile and np is not None:
                        line_pls = [profile[ys][0] for ys in range(line_y1, line_y2 + 1) if ys in profile]
                        line_prs = [profile[ys][1] for ys in range(line_y1, line_y2 + 1) if ys in profile]
                        if len(line_pls) != (line_y2 - line_y1 + 1):
                            fits = False
                            break
                        raw_cx = (np.mean(line_pls) + np.mean(line_prs)) / 2.0
                        max_drift = max(10, int(avail_w * 0.20))
                        line_cx = max(cx - max_drift, min(cx + max_drift, raw_cx))
                        for ys in range(line_y1, line_y2 + 1):
                            pl, pr = profile[ys]
                            cl_l = (line_cx - lw / 2) - pl
                            cl_r = pr - (line_cx + lw / 2)
                            min_clearance = min(min_clearance, cl_l, cl_r)
                            if cl_l < M or cl_r < M:
                                fits = False
                                break
                    elif (cx - lw / 2 < min_c + M) or (cx + lw / 2 > max_c - M):
                        fits = False
                        break
                    if not fits:
                        break

                if fits:
                    line_lens = [len(l) for l in lines]
                    var_pen = (float(np.std(line_lens)) * 0.3) if np is not None else 0.0
                    single_word_pen = sum(2.0 for l in lines if len(l.split()) == 1 and len(l) < 3)

                    # Aspect ratio & ladder penalty: penalize single-word vertical ladders only when bubble has wide aspect ratio
                    ladder_pen = 0.0
                    avg_wpl = N / max(1, k)
                    if N >= 5 and avg_wpl < min_wpl:
                        ladder_pen = 350.0 * (min_wpl - avg_wpl) + 100.0

                    # Hyphenation penalty: un-hyphenated lines preferred
                    hyphen_pen = sum(150.0 for l in lines if l.endswith("-"))

                    # Space utilization bonus: reward filling available width and height naturally
                    max_lw = max(measure_line_length(draw, l, f, f_sym) for l in lines)
                    fill_bonus = (max_lw / max(1, avail_w)) * 120.0 + (tot_h / max(1, bubble_h)) * 100.0
                    if is_vertical and y_top < (cy - tot_h // 2):
                        fill_bonus += 25.0

                    score = (fs * 100.0) + fill_bonus + (min_clearance * 2.0) - var_pen - single_word_pen - ladder_pen - hyphen_pen
                    if score > best_score:
                        best_score = score
                        best = (f, f_sym, fs, lines, lh, line_gap, y_top)

        if best and best[2] == fs and (N < 5 or (N / max(1, len(best[3]))) >= min_wpl):
            break

    fitted = (best is not None)
    if not fitted:
        # Graceful reduction: find the largest font (>= 4) that fits within height and width
        best_fs = 4
        best_lines = words
        best_tot_h = 9999
        best_f = None
        best_f_sym = None
        best_lh = 12
        best_line_gap = 1

        wrap_w = max(20, min(avail_w, int(bubble_w * 0.90)))
        for test_fs in range(8, 3, -1):
            try:
                f_t = ImageFont.truetype(font_path, test_fs)
            except Exception:
                f_t = ImageFont.load_default()
            try:
                f_s = ImageFont.truetype(sym_font_path, max(4, min(84, int(test_fs * 1.15))))
            except Exception:
                f_s = f_t
            t_lines, bad_flow = wrap_words_for_width(draw, words, f_t, f_s, wrap_w, allow_hyphenation=True)
            if bad_flow:
                continue
            sample_bb = draw.textbbox((0, 0), "Ag", font=f_t)
            t_lh = int((sample_bb[3] - sample_bb[1]) * 1.15)
            t_gap = max(1, int(test_fs * 0.18))
            t_tot_h = len(t_lines) * t_lh + max(0, len(t_lines) - 1) * t_gap
            max_lw = max(measure_line_length(draw, l, f_t, f_s) for l in t_lines)
            M_t = max(2, min(5, int(test_fs * 0.22))) if not is_box else 2
            if t_tot_h <= (bubble_h - 2 * M_t) and max_lw <= (bubble_w - 2 * M_t):
                best_fs = test_fs
                best_lines = t_lines
                best_tot_h = t_tot_h
                best_f = f_t
                best_f_sym = f_s
                best_lh = t_lh
                best_line_gap = t_gap
                break
        else:
            best_fs = 4
            try:
                best_f = ImageFont.truetype(font_path, 4)
            except Exception:
                best_f = ImageFont.load_default()
            best_f_sym = best_f
            best_lines, _ = wrap_words_for_width(draw, words, best_f, best_f_sym, wrap_w, allow_hyphenation=True)
            sample_bb = draw.textbbox((0, 0), "Ag", font=best_f)
            best_lh = int((sample_bb[3] - sample_bb[1]) * 1.15)
            best_line_gap = 1
            best_tot_h = len(best_lines) * best_lh + max(0, len(best_lines) - 1) * best_line_gap

        M_safe = max(2, int(best_fs * 0.16))
        y_top = max(min_r + M_safe, min(cy - best_tot_h // 2, max_r - M_safe - best_tot_h))
        best = (best_f, best_f_sym, best_fs, best_lines, best_lh, best_line_gap, y_top)

    if not draw_it:
        return fitted, best[2]

    f_main, f_sym, fs, lines, lh, line_gap, y_top = best
    curr_y = y_top

    for line in lines:
        line_w = measure_line_length(draw, line, f_main, f_sym)
        line_y1 = curr_y
        line_y2 = line_y1 + lh
        if profile and np is not None:
            line_pls = [profile[ys][0] for ys in range(line_y1, line_y2 + 1) if ys in profile]
            line_prs = [profile[ys][1] for ys in range(line_y1, line_y2 + 1) if ys in profile]
            if line_pls and line_prs:
                raw_cx = (np.mean(line_pls) + np.mean(line_prs)) / 2.0
                max_drift = max(10, int(avail_w * 0.20))
                line_cx = max(cx - max_drift, min(cx + max_drift, raw_cx))
            else:
                line_cx = cx
        else:
            line_cx = cx
        curr_x = int(line_cx - line_w / 2)
        draw_line_with_symbols(
            draw,
            line,
            curr_x,
            curr_y,
            fill_color,
            f_main,
            f_sym,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
        )
        curr_y += lh + line_gap

    return fitted, fs


def draw_fitted(
    draw: "ImageDraw.ImageDraw",
    text: str,
    cx: int,
    cy: int,
    sw: int,
    sh: int,
    font_path: str,
    sym_font_path: str,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
    max_fs: int = 56,
    target_lang_code: str = "en",
    bubble_type: str = "dialogue",
    container_type: str = "bubble",
    profile: Optional[Dict[int, Tuple[int, int]]] = None,
    bounds: Optional[Tuple[int, int, int, int]] = None,
    stroke_width: int = 0,
    stroke_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Renders text inside a single bubble cavity with native typography."""
    _measure_or_draw(
        draw,
        text,
        cx,
        cy,
        sw,
        sh,
        font_path,
        sym_font_path,
        fill_color=fill_color,
        draw_it=True,
        max_fs=max_fs,
        target_lang_code=target_lang_code,
        bubble_type=bubble_type,
        container_type=container_type,
        profile=profile,
        bounds=bounds,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
    )


def split_text_for_regions(
    text: str,
    regions: List[Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int]]],
) -> List[Tuple[str, Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int]]]]:
    """Splits text intelligently across multiple sub-cavities at line/sentence/clause boundaries."""
    n = len(regions)
    if n <= 1:
        return [(text, regions[0])]

    # Sort regions in manga reading order (top-to-bottom, right-to-left)
    # Primary key: vertical coordinate grouped by 60px bands; Secondary key: right-to-left (-cx)
    sorted_regions = sorted(regions, key=lambda r: (r[1] // 60, -r[0]))

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) == n:
        return list(zip(lines, sorted_regions))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?~♪\n])\s+", text) if s.strip()]
    if len(sentences) == n:
        return list(zip(sentences, sorted_regions))

    clauses = [c.strip() for c in re.split(r"(?<=[.!?,;:~♪\n])\s+", text) if c.strip()]
    if len(clauses) == n:
        return list(zip(clauses, sorted_regions))

    if n == 2 and len(lines) > 2:
        area0 = sorted_regions[0][2] * sorted_regions[0][3]
        area1 = sorted_regions[1][2] * sorted_regions[1][3]
        if area0 < area1:
            return [(lines[0], sorted_regions[0]), (" ".join(lines[1:]), sorted_regions[1])]
        else:
            return [(" ".join(lines[:-1]), sorted_regions[0]), (lines[-1], sorted_regions[1])]

    words = text.split()
    if len(words) == n:
        return list(zip(words, sorted_regions))

    if len(words) < n:
        largest = max(range(n), key=lambda i: sorted_regions[i][2] * sorted_regions[i][3])
        result = [("", r) for r in sorted_regions]
        result[largest] = (text, sorted_regions[largest])
        return result

    # Proportional area allocation
    areas = [max(1, r[2] * r[3]) for r in sorted_regions]
    total_area = sum(areas)
    word_counts = [max(1, int(len(words) * (a / total_area))) for a in areas]

    diff = len(words) - sum(word_counts)
    largest_idx = areas.index(max(areas))
    word_counts[largest_idx] += diff

    chunks = []
    idx = 0
    for count in word_counts:
        part = " ".join(words[idx:idx + count])
        chunks.append(part)
        idx += count

    return list(zip(chunks, sorted_regions))
