"""Adaptive texture-aware inpainting for floating text and unboxed dialogue."""

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from .analyzer import analyze_region_tone


def zhang_suen_thinning(binary_image: "np.ndarray") -> "np.ndarray":
    """Topological thinning producing exact 1-pixel skeletons for endpoint & tangent analysis."""
    im = (binary_image > 0).astype(np.uint8)
    prev = np.zeros_like(im)
    while True:
        # Sub-iteration 1
        p2 = np.pad(im[:-1, :], ((1, 0), (0, 0)))
        p3 = np.pad(im[:-1, 1:], ((1, 0), (0, 1)))
        p4 = np.pad(im[:, 1:], ((0, 0), (0, 1)))
        p5 = np.pad(im[1:, 1:], ((0, 1), (0, 1)))
        p6 = np.pad(im[1:, :], ((0, 1), (0, 0)))
        p7 = np.pad(im[1:, :-1], ((0, 1), (1, 0)))
        p8 = np.pad(im[:, :-1], ((0, 0), (1, 0)))
        p9 = np.pad(im[:-1, :-1], ((1, 0), (1, 0)))

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )
        m1 = (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
        im[m1] = 0

        # Sub-iteration 2
        p2 = np.pad(im[:-1, :], ((1, 0), (0, 0)))
        p3 = np.pad(im[:-1, 1:], ((1, 0), (0, 1)))
        p4 = np.pad(im[:, 1:], ((0, 0), (0, 1)))
        p5 = np.pad(im[1:, 1:], ((0, 1), (0, 1)))
        p6 = np.pad(im[1:, :], ((0, 1), (0, 0)))
        p7 = np.pad(im[1:, :-1], ((0, 1), (1, 0)))
        p8 = np.pad(im[:, :-1], ((0, 0), (1, 0)))
        p9 = np.pad(im[:-1, :-1], ((1, 0), (1, 0)))

        b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
        a = (
            ((p2 == 0) & (p3 == 1)).astype(int)
            + ((p3 == 0) & (p4 == 1)).astype(int)
            + ((p4 == 0) & (p5 == 1)).astype(int)
            + ((p5 == 0) & (p6 == 1)).astype(int)
            + ((p6 == 0) & (p7 == 1)).astype(int)
            + ((p7 == 0) & (p8 == 1)).astype(int)
            + ((p8 == 0) & (p9 == 1)).astype(int)
            + ((p9 == 0) & (p2 == 1)).astype(int)
        )
        m2 = (b >= 2) & (b <= 6) & (a == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
        im[m2] = 0

        if np.array_equal(im, prev):
            break
        prev = im.copy()
    return im.astype(np.uint8) * 255


def inpaint_floating(
    img_bgr: "np.ndarray",
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> None:
    """
    Inpaints floating/unboxed dialogue text with structural border preservation and curve reconstruction:
    1. Extended crop padding: captures characters and glows while strictly bounding text zone.
    2. Precise Panel Border Detection: accurately detects single lines, double lines with gutters, etc.
    3. Strictly Bounded Character Capture: erases only characters within text bounds, preventing overshoot.
    4. Structural Barrier Invariant: protects non-text artwork (hair, curves, body contours).
    5. Anti-aliased Cubic Bézier Reconstruction: detects severed contour lines and smoothly bridges them.
    6. Exact Panel Frame Restoration: re-stamps verified straight panel frames with original geometry.
    """
    if cv2 is None or np is None:
        return

    H, W = img_bgr.shape[:2]
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    pad_y = max(40, int(bh * 0.25))
    pad_x = max(35, int(bw * 0.35))
    x1 = max(0, xmin - pad_x)
    y1 = max(0, ymin - pad_y)
    x2 = min(W, xmax + pad_x)
    y2 = min(H, ymax + pad_y)

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return

    if len(roi.shape) == 2:
        gray_roi = roi
    elif len(roi.shape) == 3 and roi.shape[2] == 1:
        gray_roi = roi[:, :, 0]
    else:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rH, rW = gray_roi.shape

    tone_info = analyze_region_tone(gray_roi)
    is_dark_bg = tone_info["is_dark"]

    # 1. Dark element extraction (or light if dark scene)
    elements = (gray_roi > 170).astype(np.uint8) * 255 if is_dark_bg else (gray_roi < 90).astype(np.uint8) * 255

    # 2. Precise Panel Border Detection (Single vs. Double Lines with Gutters)
    panel_mask = np.zeros_like(elements)

    local_ymin = ymin - y1
    local_ymax = ymax - y1

    top_zone = elements[:max(1, local_ymin), :]
    bot_zone = elements[min(rH - 1, local_ymax):, :]

    top_prof = np.sum(top_zone > 0, axis=0) > min(10, max(2, top_zone.shape[0] // 3)) if top_zone.shape[0] > 0 else np.zeros(rW, dtype=bool)
    bot_prof = np.sum(bot_zone > 0, axis=0) > min(10, max(2, bot_zone.shape[0] // 3)) if bot_zone.shape[0] > 0 else np.zeros(rW, dtype=bool)

    # Check for horizontal panel frame line
    row_sums = np.sum(elements > 0, axis=1)
    horiz_rows = np.where(row_sums > rW * 0.65)[0]
    horiz_border_y = None
    if len(horiz_rows) > 0:
        horiz_border_y = int(np.median(horiz_rows))
        panel_mask[max(0, horiz_border_y - 2) : min(rH, horiz_border_y + 3), :] = 255
        if horiz_border_y > 35:
            bot_prof_h = np.sum(elements[max(0, horiz_border_y - 35) : horiz_border_y, :] > 0, axis=0) > 15
            bot_prof = bot_prof | bot_prof_h

    # Reconnect vertical panel frame columns across the gap
    for c in range(rW):
        if top_prof[c] and any(bot_prof[max(0, c - 3) : min(rW, c + 4)]):
            y_end = horiz_border_y if horiz_border_y is not None else rH
            panel_mask[:y_end, c] = 255

    # 3. Strictly Bounded Character Detection (Prevents overshoot onto stickers/props)
    non_panel = elements.copy()
    non_panel[panel_mask > 0] = 0

    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(non_panel, connectivity=8)
    char_mask = np.zeros_like(elements)
    col_cx = (xmin + xmax) // 2 - x1
    local_ymin = ymin - y1
    local_ymax = ymax - y1

    for l in range(1, n_cc):
        w = stats[l, cv2.CC_STAT_WIDTH]
        h = stats[l, cv2.CC_STAT_HEIGHT]
        a = stats[l, cv2.CC_STAT_AREA]
        clx = stats[l, cv2.CC_STAT_LEFT]
        cly = stats[l, cv2.CC_STAT_TOP]
        cx = clx + w / 2

        # Strict vertical boundary: Never overshoot beyond where Japanese text ends
        if cly < local_ymin - 25 or (cly + h) > local_ymax + 18:
            continue

        touches_edge = (clx <= 1 or cly <= 1 or clx + w >= rW - 1 or cly + h >= rH - 1)
        if touches_edge and (a > 600 or w > rW * 0.50 or h > rH * 0.40):
            continue

        max_dist_x = max(int(bw * 0.55), 18)
        if abs(cx - col_cx) <= max_dist_x and 10 <= a <= 3500 and w <= rW * 0.65 and h <= rH * 0.45:
            char_mask[labels == l] = 255

    if char_mask.max() == 0:
        return

    # 4. Total Inpaint Mask: Text strokes + glowing halo, bounded away from structural lines
    other_dark = (elements > 0) & (char_mask == 0) & (panel_mask == 0)
    prot_fig = cv2.dilate(other_dark.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dil_chars = cv2.dilate(char_mask, k_halo)
    # Strictly clamp halo vertically so inpaint never expands down onto adjacent props/stickers
    dil_chars[min(rH, local_ymax + 20) :, :] = 0
    dil_chars[: max(0, local_ymin - 25), :] = 0

    inpaint_m = (dil_chars > 0) & (prot_fig == 0)
    inpaint_m_u8 = inpaint_m.astype(np.uint8) * 255

    if inpaint_m_u8.max() == 0:
        return

    # Inpaint text & halo with texture synthesis
    cleaned = cv2.inpaint(roi, inpaint_m_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 5. Post-Inpaint Line Reconstruction (Anti-aliased cubic Bézier splines for curvy contours)
    clean_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    dark135 = (clean_gray < 135).astype(np.uint8) * 255
    if panel_mask.max() > 0:
        dark135[panel_mask > 0] = 0

    num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(dark135)
    line_mask = np.zeros_like(dark135)
    for l in range(1, num_cc):
        a = stats_cc[l, cv2.CC_STAT_AREA]
        w = stats_cc[l, cv2.CC_STAT_WIDTH]
        h = stats_cc[l, cv2.CC_STAT_HEIGHT]
        length = max(w, h)
        comp_pixels = clean_gray[labels_cc == l]
        has_dark_core = (comp_pixels < 100).any()
        if a >= 25 and length >= 12 and has_dark_core:
            line_mask[labels_cc == l] = 255

    skel = zhang_suen_thinning(line_mask)
    k_end = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint16)
    conv = cv2.filter2D((skel > 0).astype(np.uint16), -1, k_end)
    endpoints = np.argwhere(conv == 11)

    # Keep only endpoints within the severed text zone
    dil_inp = cv2.dilate(inpaint_m_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    severed_endpoints = [pt for pt in endpoints if dil_inp[pt[0], pt[1]] > 0]

    def get_tangent(pt, skel_img, max_steps=12):
        y, x = pt
        visited = set([(y, x)])
        curr = (y, x)
        path = [curr]
        for _ in range(max_steps):
            cy, cx = curr
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rH and 0 <= nx < rW and skel_img[ny, nx] > 0 and (ny, nx) not in visited:
                        neighbors.append((ny, nx))
            if not neighbors:
                break
            curr = neighbors[0]
            visited.add(curr)
            path.append(curr)
        if len(path) < 3:
            return None
        vy = path[0][0] - path[-1][0]
        vx = path[0][1] - path[-1][1]
        mag = np.sqrt(vx**2 + vy**2)
        return (vx / mag, vy / mag) if mag > 0 else None

    dist_map = cv2.distanceTransform(line_mask, cv2.DIST_L2, 5)

    def get_stroke_width(pt, skel_img, dist_m, steps=8):
        y, x = pt
        visited = set([(y, x)])
        curr = (y, x)
        widths = []
        for _ in range(steps):
            cy, cx = curr
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < rH and 0 <= nx < rW and skel_img[ny, nx] > 0 and (ny, nx) not in visited:
                        neighbors.append((ny, nx))
            if not neighbors:
                break
            curr = neighbors[0]
            visited.add(curr)
            widths.append(dist_m[curr[0], curr[1]] * 2.0)
        return float(np.median(widths)) if widths else 2.0

    ep_info = []
    for y, x in severed_endpoints:
        if x <= 2 or y <= 2 or x >= rW - 3 or y >= rH - 3:
            continue
        t = get_tangent((y, x), skel)
        if t is not None:
            w = get_stroke_width((y, x), skel, dist_map)
            ep_info.append(((x, y), t, w))

    # Pairwise matching based on collinearity, distance, and tangent alignment
    matched = []
    used = set()
    for i in range(len(ep_info)):
        if i in used:
            continue
        p1, t1, w1 = ep_info[i]
        best_j, best_s = None, -1e9
        for j in range(len(ep_info)):
            if i == j or j in used:
                continue
            p2, t2, w2 = ep_info[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx**2 + dy**2)
            if 12 <= dist <= 120:
                d_norm = (dx / dist, dy / dist)
                align1 = t1[0] * d_norm[0] + t1[1] * d_norm[1]
                align2 = t2[0] * (-d_norm[0]) + t2[1] * (-d_norm[1])
                if align1 > 0.20 and align2 > 0.20:
                    score = (align1 + align2) * 50 - dist
                    if score > best_s:
                        best_s, best_j = score, j
        if best_j is not None:
            p2, t2, w2 = ep_info[best_j]
            d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            matched.append((p1, p2, t1, t2, w1, w2, d))
            used.add(i)
            used.add(best_j)

    # Render smooth anti-aliased cubic Bézier splines with subpixel fixed-point precision
    shift = 4
    sub = 1 << shift
    for p1, p2, t1, t2, w1, w2, d in matched:
        scale = d / 3.0
        c1 = (p1[0] + t1[0] * scale, p1[1] + t1[1] * scale)
        c2 = (p2[0] + t2[0] * scale, p2[1] + t2[1] * scale)
        pts = []
        for t in np.linspace(0, 1, 100):
            bx = (1 - t) ** 3 * p1[0] + 3 * (1 - t) ** 2 * t * c1[0] + 3 * (1 - t) * t**2 * c2[0] + t**3 * p2[0]
            by = (1 - t) ** 3 * p1[1] + 3 * (1 - t) ** 2 * t * c1[1] + 3 * (1 - t) * t**2 * c2[1] + t**3 * p2[1]
            pts.append([int(round(bx * sub)), int(round(by * sub))])
        pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        avg_w = (w1 + w2) / 2.0
        line_th = max(1, int(round(avg_w / 2.0)))
        cv2.polylines(cleaned, [pts], isClosed=False, color=(0, 0, 0), thickness=line_th, lineType=cv2.LINE_AA, shift=shift)

    # 6. Re-stamp all verified panel border lines (single, double, with gutters)
    if panel_mask.max() > 0:
        cleaned[panel_mask > 0] = [0, 0, 0]

    img_bgr[y1:y2, x1:x2] = cleaned
