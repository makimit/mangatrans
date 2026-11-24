"""Conjoined bubble lobe segmentation, constriction detection, and capacity checks."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


def segment_cavity_lobes(
    cavity: "np.ndarray",
    min_c: int,
    min_r: int,
    max_c: int,
    max_r: int,
    x1: int,
    y1: int,
) -> List[Tuple["np.ndarray", int, int, int, int]]:
    """
    Detects waists/constrictions in conjoined multi-lobed speech bubbles and segments them into individual lobes.
    Supports both vertically stacked lobes and horizontally connected lobes.
    """
    if cavity is None or cavity.size == 0 or np is None:
        return []

    ys_c, xs_c = np.where(cavity > 0)
    if len(ys_c) == 0:
        return []

    y_min_i = int(np.min(ys_c))
    y_max_i = int(np.max(ys_c))
    x_min_i = int(np.min(xs_c))
    x_max_i = int(np.max(xs_c))

    # --- 1. Vertical axis waist detection (top/bottom lobes) ---
    row_widths = []
    y_indices = []

    for y_idx in range(y_min_i, y_max_i + 1):
        xs_row = xs_c[ys_c == y_idx]
        if len(xs_row) > 0:
            w = int(np.max(xs_row) - np.min(xs_row))
            row_widths.append(w)
            y_indices.append(y_idx)

    v_valleys = []
    if len(row_widths) >= 40:
        ksize = max(7, len(row_widths) // 10)
        smoothed = np.convolve(row_widths, np.ones(ksize) / ksize, mode="same")
        raw_valleys = []
        for i in range(ksize, len(smoothed) - ksize):
            if smoothed[i] == min(smoothed[i - ksize:i + ksize + 1]):
                left_max = max(smoothed[:i]) if i > 0 else smoothed[i]
                right_max = max(smoothed[i:]) if i < len(smoothed) - 1 else smoothed[i]
                if smoothed[i] < 0.40 * min(left_max, right_max) and min(left_max, right_max) > 25:
                    raw_valleys.append((y_indices[i], smoothed[i]))

        # Filter close valleys and ensure waist is not within 60px of cavity boundary
        filtered = []
        for vy, val in raw_valleys:
            if not filtered:
                filtered.append((vy, val))
            else:
                prev_y, prev_val = filtered[-1]
                if abs(vy - prev_y) < 35:
                    if val < prev_val:
                        filtered[-1] = (vy, val)
                else:
                    filtered.append((vy, val))
        v_valleys = [
            vy for vy, _ in filtered
            if (vy - y_min_i >= 60 and (y_max_i + 1) - vy >= 60)
        ]

    # Check for multi-lobe waists using contour convexity defects (handles offset vertical and horizontal lobes)
    if not v_valleys and cv2 is not None and len(ys_c) > 400:
        c_h = y_max_i - y_min_i
        c_w = x_max_i - x_min_i
        c_cx = (x_min_i + x_max_i) / 2.0
        c_cy = (y_min_i + y_max_i) / 2.0
        contours, _ = cv2.findContours(cavity.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt, returnPoints=False)
            if hull is not None and len(hull) >= 4:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is not None:
                    defects = defects.reshape(-1, 4)
                    deep_defects = []
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i]
                        depth = float(d) / 256.0
                        far_pt = cnt[f][0]
                        if depth >= 15.0:
                            deep_defects.append((far_pt[0], far_pt[1], depth))

                    if len(deep_defects) >= 2:
                        deep_defects.sort(key=lambda d: d[2], reverse=True)
                        for i in range(len(deep_defects)):
                            x_a, y_a, d_a = deep_defects[i]
                            for j in range(i + 1, len(deep_defects)):
                                x_b, y_b, d_b = deep_defects[j]
                                dx = abs(x_a - x_b)
                                dy = abs(y_a - y_b)

                                # Case A: Horizontal split (left and right lobes, opposing defects top and bottom)
                                if c_w >= c_h * 0.70 and dy > 1.2 * dx and dy >= c_h * 0.18 and ((y_a - c_cy) * (y_b - c_cy) < 0):
                                    mid_x = int((x_a + x_b) // 2)
                                    if (mid_x - x_min_i) >= 75 and (x_max_i - mid_x) >= 75:
                                        y_bot_cut = max(y_a, y_b)
                                        m_aux = np.zeros_like(cavity)
                                        m_main = cavity.copy()
                                        if (y_bot_cut - y_min_i) < 0.70 * c_h:
                                            m_aux[:y_bot_cut + 8, mid_x:] = cavity[:y_bot_cut + 8, mid_x:]
                                            m_main[:y_bot_cut + 8, mid_x:] = 0
                                        else:
                                            m_aux[:, mid_x:] = cavity[:, mid_x:]
                                            m_main[:, mid_x:] = 0
                                        y1_m, x1_m = np.where(m_main > 0)
                                        y2_m, x2_m = np.where(m_aux > 0)
                                        if (
                                            len(y1_m) > 150 and len(y2_m) > 150
                                            and (np.max(y1_m) - np.min(y1_m) >= 60)
                                            and (np.max(y2_m) - np.min(y2_m) >= 60)
                                            and (np.max(x1_m) - np.min(x1_m) >= 45)
                                            and (np.max(x2_m) - np.min(x2_m) >= 45)
                                        ):
                                            return [
                                                (m_main, int(np.min(x1_m)) + x1, int(np.min(y1_m)) + y1, int(np.max(x1_m)) + x1, int(np.max(y1_m)) + y1),
                                                (m_aux, int(np.min(x2_m)) + x1, int(np.min(y2_m)) + y1, int(np.max(x2_m)) + x1, int(np.max(y2_m)) + y1),
                                            ]

                                # Case B: Vertical split (top and bottom lobes, opposing defects left and right)
                                elif dx >= 0.7 * dy and dx >= c_w * 0.18 and ((x_a - c_cx) * (x_b - c_cx) < 0):
                                    mid_y = int((y_a + y_b) // 2)
                                    if (mid_y - y_min_i) >= 60 and (y_max_i - mid_y) >= 60:
                                        m1 = np.zeros_like(cavity)
                                        m1[:mid_y, :] = cavity[:mid_y, :]
                                        m2 = np.zeros_like(cavity)
                                        m2[mid_y:, :] = cavity[mid_y:, :]
                                        y1_m, x1_m = np.where(m1 > 0)
                                        y2_m, x2_m = np.where(m2 > 0)
                                        if (
                                            len(y1_m) > 150 and len(y2_m) > 150
                                            and (np.max(y1_m) - np.min(y1_m) >= 60)
                                            and (np.max(y2_m) - np.min(y2_m) >= 60)
                                            and (np.max(x1_m) - np.min(x1_m) >= 45)
                                            and (np.max(x2_m) - np.min(x2_m) >= 45)
                                        ):
                                            return [
                                                (m1, int(np.min(x1_m)) + x1, int(np.min(y1_m)) + y1, int(np.max(x1_m)) + x1, int(np.max(y1_m)) + y1),
                                                (m2, int(np.min(x2_m)) + x1, int(np.min(y2_m)) + y1, int(np.max(x2_m)) + x1, int(np.max(y2_m)) + y1),
                                            ]

    if v_valleys:
        lobes = []
        split_ys = [y_min_i] + v_valleys + [y_max_i + 1]
        for s_idx in range(len(split_ys) - 1):
            y_start = split_ys[s_idx]
            y_end = split_ys[s_idx + 1]
            if y_end - y_start < 60:
                continue
            lobe_mask = np.zeros_like(cavity)
            lobe_mask[y_start:y_end, :] = cavity[y_start:y_end, :]
            l_ys, l_xs = np.where(lobe_mask > 0)
            if len(l_ys) > 200:
                l_min_c = int(np.min(l_xs)) + x1
                l_max_c = int(np.max(l_xs)) + x1
                l_min_r = int(np.min(l_ys)) + y1
                l_max_r = int(np.max(l_ys)) + y1
                if (l_max_r - l_min_r >= 60) and (l_max_c - l_min_c >= 45):
                    lobes.append((lobe_mask, l_min_c, l_min_r, l_max_c, l_max_r))
        if len(lobes) > 1 and len(lobes) == len(split_ys) - 1:
            return lobes

    return [(cavity, min_c, min_r, max_c, max_r)]


def is_lobe_viable_for_text(
    lobe_w: int,
    lobe_h: int,
    word_count: int = 1,
    min_pt: int = 12,
) -> bool:
    """
    Checks if a lobe's dimensions can comfortably accommodate legible English text (>= min_pt).
    If a conjoined sub-lobe is too tiny (e.g. narrow tail or short taper), it returns False
    so the pipeline merges the text into the primary bubble and clears the tiny lobe.
    """
    if lobe_w < 45 or lobe_h < 60:
        return False
    # Approximate text area required for min_pt readable font
    est_char_h = int(min_pt * 1.3)
    est_char_w = int(min_pt * 0.7)
    needed_h = est_char_h + 6
    needed_w = max(40, word_count * est_char_w * 4)
    if lobe_h < needed_h or (lobe_w * lobe_h) < 1800:
        return False
    return True


def _clamp_region_x(region: Tuple[Any, ...], min_x: Optional[int] = None, max_x: Optional[int] = None) -> Tuple[Any, ...]:
    cx, cy, sw, sh, text_color = region[0], region[1], region[2], region[3], region[4]
    profile = region[5] if len(region) > 5 else None
    bounds = region[6] if len(region) > 6 else None
    stroke_w = region[7] if len(region) > 7 else 0
    stroke_c = region[8] if len(region) > 8 else None

    if profile:
        new_prof = {}
        for y, (pl, pr) in profile.items():
            n_pl = max(pl, min_x) if min_x is not None else pl
            n_pr = min(pr, max_x) if max_x is not None else pr
            if n_pr - n_pl >= 15:
                new_prof[y] = (n_pl, n_pr)
        if new_prof:
            p_ys = list(new_prof.keys())
            act_min_y = min(p_ys)
            act_max_y = max(p_ys)
            act_min_x = min(new_prof[y][0] for y in p_ys)
            act_max_x = max(new_prof[y][1] for y in p_ys)
            raw_w = act_max_x - act_min_x
            raw_h = act_max_y - act_min_y
            new_sw = max(20, min(int(raw_w * 0.82), raw_w - 14))
            new_sh = max(20, min(int(raw_h * 0.84), raw_h - 14))
            if np is not None:
                new_cx = int(np.mean([(pl + pr) / 2.0 for pl, pr in new_prof.values()]))
                new_cy = int(np.mean(p_ys))
            else:
                new_cx = (act_min_x + act_max_x) // 2
                new_cy = (act_min_y + act_max_y) // 2
            new_bounds = (act_min_x, act_min_y, act_max_x, act_max_y)
            return (new_cx, new_cy, new_sw, new_sh, text_color, new_prof, new_bounds, stroke_w, stroke_c)

    if bounds:
        b_min_x, b_min_y, b_max_x, b_max_y = bounds
        n_min_x = max(b_min_x, min_x) if min_x is not None else b_min_x
        n_max_x = min(b_max_x, max_x) if max_x is not None else b_max_x
        if n_max_x > n_min_x:
            raw_w = n_max_x - n_min_x
            raw_h = b_max_y - b_min_y
            new_sw = max(20, min(int(raw_w * 0.82), raw_w - 14))
            new_cx = (n_min_x + n_max_x) // 2
            new_bounds = (n_min_x, b_min_y, n_max_x, b_max_y)
            return (new_cx, cy, new_sw, sh, text_color, profile, new_bounds, stroke_w, stroke_c)

    return region


def _clamp_region_y(region: Tuple[Any, ...], min_y: Optional[int] = None, max_y: Optional[int] = None) -> Tuple[Any, ...]:
    cx, cy, sw, sh, text_color = region[0], region[1], region[2], region[3], region[4]
    profile = region[5] if len(region) > 5 else None
    bounds = region[6] if len(region) > 6 else None
    stroke_w = region[7] if len(region) > 7 else 0
    stroke_c = region[8] if len(region) > 8 else None

    if profile:
        new_prof = {}
        for y, (pl, pr) in profile.items():
            if (min_y is None or y >= min_y) and (max_y is None or y <= max_y):
                if pr - pl >= 15:
                    new_prof[y] = (pl, pr)
        if new_prof:
            p_ys = list(new_prof.keys())
            act_min_y = min(p_ys)
            act_max_y = max(p_ys)
            act_min_x = min(new_prof[y][0] for y in p_ys)
            act_max_x = max(new_prof[y][1] for y in p_ys)
            raw_w = act_max_x - act_min_x
            raw_h = act_max_y - act_min_y
            new_sw = max(20, min(int(raw_w * 0.82), raw_w - 14))
            new_sh = max(20, min(int(raw_h * 0.84), raw_h - 14))
            if np is not None:
                new_cx = int(np.mean([(pl + pr) / 2.0 for pl, pr in new_prof.values()]))
                new_cy = int(np.mean(p_ys))
            else:
                new_cx = (act_min_x + act_max_x) // 2
                new_cy = (act_min_y + act_max_y) // 2
            new_bounds = (act_min_x, act_min_y, act_max_x, act_max_y)
            return (new_cx, new_cy, new_sw, new_sh, text_color, new_prof, new_bounds, stroke_w, stroke_c)

    if bounds:
        b_min_x, b_min_y, b_max_x, b_max_y = bounds
        n_min_y = max(b_min_y, min_y) if min_y is not None else b_min_y
        n_max_y = min(b_max_y, max_y) if max_y is not None else b_max_y
        if n_max_y > n_min_y:
            raw_h = n_max_y - n_min_y
            new_sh = max(20, min(int(raw_h * 0.84), raw_h - 14))
            new_cy = (n_min_y + n_max_y) // 2
            new_bounds = (b_min_x, n_min_y, b_max_x, n_max_y)
            return (cx, new_cy, sw, new_sh, text_color, profile, new_bounds, stroke_w, stroke_c)

    return region


def partition_overlapping_regions(
    centroid_data: List[Tuple[Dict[str, Any], List[Any]]],
    H: int,
    W: int,
) -> None:
    """
    Detects when multiple neighboring or conjoined speech bubbles share overlapping cavity regions
    and partitions the cavity envelopes along their mutual boundary so that dialogue text never collides.
    """
    if not centroid_data:
        return

    # Run up to 3 passes to handle multi-bubble conjoined chains
    for _ in range(3):
        changed = False
        for i in range(len(centroid_data)):
            item_i, regions_i = centroid_data[i]
            box_i = item_i.get("box_2d")
            if not box_i or len(box_i) != 4 or not regions_i:
                continue
            y1_i, x1_i, y2_i, x2_i = [
                int(box_i[0] * H / 1000),
                int(box_i[1] * W / 1000),
                int(box_i[2] * H / 1000),
                int(box_i[3] * W / 1000),
            ]
            cx_i = (x1_i + x2_i) / 2.0
            cy_i = (y1_i + y2_i) / 2.0

            for j in range(i + 1, len(centroid_data)):
                item_j, regions_j = centroid_data[j]
                box_j = item_j.get("box_2d")
                if not box_j or len(box_j) != 4 or not regions_j:
                    continue
                y1_j, x1_j, y2_j, x2_j = [
                    int(box_j[0] * H / 1000),
                    int(box_j[1] * W / 1000),
                    int(box_j[2] * H / 1000),
                    int(box_j[3] * W / 1000),
                ]
                cx_j = (x1_j + x2_j) / 2.0
                cy_j = (y1_j + y2_j) / 2.0

                for reg_idx_i, r_i in enumerate(regions_i):
                    b_i = r_i[6]
                    if not b_i:
                        continue
                    for reg_idx_j, r_j in enumerate(regions_j):
                        b_j = r_j[6]
                        if not b_j:
                            continue

                        ix1 = max(b_i[0], b_j[0])
                        iy1 = max(b_i[1], b_j[1])
                        ix2 = min(b_i[2], b_j[2])
                        iy2 = min(b_i[3], b_j[3])

                        if ix2 > ix1 + 10 and iy2 > iy1 + 10:
                            p_i = r_i[5]
                            p_j = r_j[5]
                            has_overlap = True
                            if p_i and p_j:
                                common_ys = set(p_i.keys()) & set(p_j.keys())
                                has_overlap = any(
                                    max(p_i[y][0], p_j[y][0]) < min(p_i[y][1], p_j[y][1])
                                    for y in common_ys
                                )

                            if has_overlap:
                                c_w = max(b_i[2] - b_i[0], b_j[2] - b_j[0])
                                c_h = max(b_i[3] - b_i[1], b_j[3] - b_j[1])
                                dx = cx_j - cx_i
                                dy = cy_j - cy_i

                                # Determine partition axis primarily from relative placement of the two dialogue texts:
                                # - If texts are side-by-side horizontally (abs(dx) > abs(dy)), partition along X (vertical dividing line).
                                # - If texts are stacked vertically (abs(dy) > abs(dx)), partition along Y (horizontal dividing line).
                                if abs(dx) > 1.2 * abs(dy):
                                    partition_axis = "x"
                                elif abs(dy) > 1.2 * abs(dx):
                                    partition_axis = "y"
                                elif abs(dx) > abs(dy):
                                    partition_axis = "x"
                                else:
                                    partition_axis = "y"

                                if partition_axis == "x":
                                    cavity_mid_x = int((min(b_i[0], b_j[0]) + max(b_i[2], b_j[2])) // 2)
                                    if dx > 0 or (dx == 0 and x1_i <= x1_j):
                                        split_x = int((x2_i + x1_j) // 2) if x2_i <= x1_j else cavity_mid_x
                                        regions_i[reg_idx_i] = _clamp_region_x(r_i, max_x=split_x - 1)
                                        regions_j[reg_idx_j] = _clamp_region_x(r_j, min_x=split_x + 1)
                                    else:
                                        split_x = int((x2_j + x1_i) // 2) if x2_j <= x1_i else cavity_mid_x
                                        regions_j[reg_idx_j] = _clamp_region_x(r_j, max_x=split_x - 1)
                                        regions_i[reg_idx_i] = _clamp_region_x(r_i, min_x=split_x + 1)
                                else:
                                    cavity_mid_y = int((min(b_i[1], b_j[1]) + max(b_i[3], b_j[3])) // 2)
                                    if dy > 0 or (dy == 0 and y1_i <= y1_j):
                                        split_y = int((y2_i + y1_j) // 2) if y2_i <= y1_j else cavity_mid_y
                                        regions_i[reg_idx_i] = _clamp_region_y(r_i, max_y=split_y - 1)
                                        regions_j[reg_idx_j] = _clamp_region_y(r_j, min_y=split_y + 1)
                                    else:
                                        split_y = int((y2_j + y1_i) // 2) if y2_j <= y1_i else cavity_mid_y
                                        regions_j[reg_idx_j] = _clamp_region_y(r_j, max_y=split_y - 1)
                                        regions_i[reg_idx_i] = _clamp_region_y(r_i, min_y=split_y + 1)
                                changed = True
        if not changed:
            break

