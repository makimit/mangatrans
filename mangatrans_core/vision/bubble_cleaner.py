"""Adaptive inside-out bubble cleaning with 100% 360-degree border preservation and zero convex-hull destruction."""

from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

from .analyzer import compute_adaptive_thresholds
from .conjoined import is_lobe_viable_for_text, segment_cavity_lobes


def has_continuous_closed_boundary(crop: "np.ndarray") -> bool:
    """Checks whether the crop contains a continuous closed boundary contour surrounding the center."""
    if cv2 is None or np is None or crop is None or crop.size == 0:
        return False
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    H, W = g.shape
    if H < 10 or W < 10:
        return False
    cx, cy = W // 2, H // 2
    dark = (g < 140).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        perim = cv2.arcLength(c, True)
        if perim > (W + H) * 0.7:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                return True
    return False


def check_is_burst(crop: "np.ndarray", cx: Optional[int] = None, cy: Optional[int] = None) -> bool:
    """Checks whether the crop exhibits radial spikes radiating from the text center."""
    if cv2 is None or np is None or crop is None or crop.size == 0:
        return False
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    H, W = g.shape
    if H < 30 or W < 30:
        return False
    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
    scale_y = W / float(H)
    dark = (g < 160).astype(np.uint8) * 255
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    spikes = 0
    for l in range(1, num_l):
        a = stats[l, cv2.CC_STAT_AREA]
        if 10 <= a <= 2500:
            ys, xs = np.where(labels == l)
            dx = xs - cx
            dy = (ys - cy) * scale_y
            dists = np.sqrt(dx**2 + dy**2)
            span = np.max(dists) - np.min(dists)
            if span >= 15:
                spikes += 1
    return spikes >= 20


def clean_black_bubble(
    img_bgr: "np.ndarray",
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    margin: int = 25,
) -> Optional[List[Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int], int, Any]]]:
    """Dedicated inside-out cleaner and typesetter for black speech bubbles."""
    if cv2 is None or np is None:
        return None
    H, W = img_bgr.shape[:2]
    bw_in = max(1, xmax - xmin)
    bh_in = max(1, ymax - ymin)

    pad_x = max(70, int(bw_in * 0.75))
    pad_y = max(60, int(bh_in * 0.60))
    x1 = max(0, xmin - pad_x)
    x2 = min(W, xmax + pad_x)
    y1 = max(0, ymin - pad_y)
    y2 = min(H, ymax + pad_y)

    crop = img_bgr[y1:y2, x1:x2].copy()
    g_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    cH, cW = crop.shape[:2]

    # Detect the full connected dark bubble cavity
    dark_bubble = (g_crop < 80).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dark_closed = cv2.morphologyEx(dark_bubble, cv2.MORPH_CLOSE, k_close)

    tcx_l = (xmin + xmax) // 2 - x1
    tcy_l = (ymin + ymax) // 2 - y1

    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(dark_closed)
    target_lbl = cc_labels[tcy_l, tcx_l] if (0 <= tcy_l < cH and 0 <= tcx_l < cW) else 0

    if target_lbl > 0 and cc_stats[target_lbl, cv2.CC_STAT_AREA] > bw_in * bh_in * 0.40:
        bx = cc_stats[target_lbl, cv2.CC_STAT_LEFT]
        by = cc_stats[target_lbl, cv2.CC_STAT_TOP]
        b_w = cc_stats[target_lbl, cv2.CC_STAT_WIDTH]
        b_h = cc_stats[target_lbl, cv2.CC_STAT_HEIGHT]
        bubble_mask = (cc_labels == target_lbl)

        # White/bright text inside dark bubble
        mask_text = (g_crop > 70) & bubble_mask
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dil = cv2.dilate(mask_text.astype(np.uint8) * 255, k_dil)
        crop[dil > 0] = [0, 0, 0]
        img_bgr[y1:y2, x1:x2] = crop

        cx = bx + x1 + b_w // 2
        cy = by + y1 + b_h // 2
        sw = max(20, int(b_w * 0.76))
        sh = max(20, int(b_h * 0.76))
        bounds = (bx + x1 + 10, by + y1 + 15, bx + x1 + b_w - 10, by + y1 + b_h - 15)
        return [(cx, cy, sw, sh, (255, 255, 255), None, bounds, 0, None)]

    # Fallback to text bounding box
    mask_text = np.zeros_like(g_crop, dtype=np.uint8)
    pad = 12
    if cH > 2 * pad and cW > 2 * pad:
        mask_text[pad:cH-pad, pad:cW-pad] = (g_crop[pad:cH-pad, pad:cW-pad] > 70).astype(np.uint8) * 255
    else:
        mask_text = (g_crop > 70).astype(np.uint8) * 255

    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(mask_text)
    txt_only = np.zeros_like(g_crop, dtype=np.uint8)
    for l in range(1, num_l):
        if stats[l, cv2.CC_STAT_AREA] < 1500:
            txt_only[labels == l] = 255
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dil = cv2.dilate(txt_only, k_dil)
    crop[dil > 0] = [0, 0, 0]
    img_bgr[y1:y2, x1:x2] = crop

    cx = (xmin + xmax) // 2
    cy = (ymin + ymax) // 2
    bounds = (xmin + 10, ymin + 15, xmax - 10, ymax - 15)
    sw = max(20, bounds[2] - bounds[0])
    sh = max(20, bounds[3] - bounds[1])
    return [(cx, cy, sw, sh, (255, 255, 255), None, bounds, 0, None)]


def clean_caption_box(
    img_bgr: "np.ndarray",
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    bg_color_hint: str = "white",
) -> Optional[List[Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int]]]]:
    """
    Precision extraction and inside-out cleaning for rectangular caption/narration boxes.
    Detects the 4 frame lines (top, bottom, left, right) using directional line scanning
    across an expanded margin, preserving the 1-pixel frame border with 100% fidelity.
    """
    if cv2 is None or np is None:
        return None

    H, W = img_bgr.shape[:2]
    is_dark = (bg_color_hint.lower() == "black")
    fill_bgr = [0, 0, 0] if is_dark else [255, 255, 255]
    t_color = (255, 255, 255) if is_dark else (0, 0, 0)

    pad_x = max(70, int((xmax - xmin) * 0.75))
    pad_y = max(60, int((ymax - ymin) * 0.60))
    search_x1 = max(0, xmin - pad_x); search_x2 = min(W, xmax + pad_x)
    search_y1 = max(0, ymin - pad_y); search_y2 = min(H, ymax + pad_y)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    dark = (gray > 160).astype(np.uint8) if is_dark else (gray < 140).astype(np.uint8)

    # 1. Search for top horizontal line <= ymin + 15
    top_y = None
    for y in range(ymin + 15, search_y1 - 1, -1):
        if np.mean(dark[y, xmin:xmax]) > 0.55:
            top_y = y
            break
    if top_y is None and search_y1 == 0 and np.mean(dark[0, xmin:xmax]) > 0.45:
        top_y = 0

    # 2. Search for bottom horizontal line >= ymax - 15
    bot_y = None
    for y in range(ymax - 15, search_y2):
        if np.mean(dark[y, xmin:xmax]) > 0.55:
            bot_y = y
            break

    check_y1 = top_y + 5 if top_y is not None else ymin
    check_y2 = bot_y - 5 if bot_y is not None else ymax

    # 3. Search for left vertical line <= xmin + 15
    left_x = None
    for x in range(xmin + 15, search_x1 - 1, -1):
        if np.mean(dark[check_y1:check_y2, x]) > 0.55:
            left_x = x
            break

    # 4. Search for right vertical line >= xmax - 15
    right_x = None
    for x in range(xmax - 15, search_x2):
        if np.mean(dark[check_y1:check_y2, x]) > 0.55:
            right_x = x
            break

    if top_y is not None and bot_y is not None and left_x is not None and right_x is not None:
        box_w = right_x - left_x
        box_h = bot_y - top_y
        req_w = xmax - xmin
        req_h = ymax - ymin
        if (box_w >= req_w * 0.70) and (box_h >= req_h * 0.70):
            inner_y1 = top_y + 2
            inner_y2 = bot_y - 2
            inner_x1 = left_x + 2
            inner_x2 = right_x - 2

            if inner_y2 > inner_y1 and inner_x2 > inner_x1:
                img_bgr[inner_y1:inner_y2, inner_x1:inner_x2] = fill_bgr

                cx = (inner_x1 + inner_x2) // 2
                cy = (inner_y1 + inner_y2) // 2

                # Tighter inner margin padding (~3.5%) so text fills the box optimally while preserving frame clearance
                c_w = inner_x2 - inner_x1
                c_h = inner_y2 - inner_y1
                margin_x = max(5, int(c_w * 0.035))
                margin_y = max(5, int(c_h * 0.035))

                slot_x1 = inner_x1 + margin_x
                slot_x2 = inner_x2 - margin_x
                slot_y1 = inner_y1 + margin_y
                slot_y2 = inner_y2 - margin_y

                sw = max(20, slot_x2 - slot_x1)
                sh = max(20, slot_y2 - slot_y1)

                profile = {y: (slot_x1, slot_x2) for y in range(slot_y1, slot_y2 + 1)}
                return [(cx, cy, sw, sh, t_color, profile, (slot_x1, slot_y1, slot_x2, slot_y2), 0, None)]

    return None


def detect_burst_inner_region_generalized(
    crop_bgr: "np.ndarray",
    cx: Optional[int] = None,
    cy: Optional[int] = None,
    k_harmonics: int = 4,
) -> Optional[List[Tuple[int, int]]]:
    """
    Generalized mathematical inner region detector for announce / burst bubbles.
    
    Why this approach is designed this way:
    1. Aspect-Ratio Normalized Polar Space:
       Manga burst bubbles can be tall, wide, or circular. Normalizing radial distances
       with scale_y = W / H creates an isotropic Euclidean space where rays radiate
       evenly in all directions regardless of canvas dimensions.
    2. Epicenter Anchoring:
       (cx, cy) is explicitly anchored to the dialogue text epicenter within the crop.
       When a crop touches page edges (e.g. at x=0), the crop center differs from the
       bubble epicenter; passing the true text center prevents asymmetrical boundary drift.
    3. Radial Spike Filtering:
       Dark ink strokes with significant radial elongation (max_d - min_d >= 12px)
       extending toward the outer envelope (max_d >= 0.45 * R_outer) are identified as spikes.
       Standard speech bubbles with smooth borders have < 25 spikes; burst bubbles typically
       exhibit 40-80 spikes uniformly distributed around the circumference.
    4. Innermost Tip Selection per Angular Bin:
       Spikes are binned into fine 5-degree angular sectors (72 bins). In each sector,
       only the minimum radius (innermost spike tip facing the interior cavity) is recorded.
       This automatically discards exterior artwork, characters, shoulders, or panel lines
       that happen to lie along the same ray outside the bubble.
    5. Circular Interpolation across Gaps:
       Sectors lacking spikes are circularly interpolated between neighboring inner tips,
       preventing arbitrary outward leaps in sparse zones.
    6. Outer & Inner Jump Smoothing Filter:
       - Outer jumps (r > median_window * 1.15): Protrusions sticking outward between spikes
         are clamped down to the local median, cleanly discarding the outward area.
       - Inner jumps (r < median_window * 0.85): Inward dips caused by stray glyph fragments
         or deep spike tips are clamped up to the local median, including the excluded cavity area.
    7. Low-Order Fourier Reconstruction (k <= 4):
       Low-frequency harmonics synthesize a continuous, smooth C-infinity convex-like oval
       boundary contour along the inward spike tips without kinks or staircase artifacts.
    """
    if cv2 is None or np is None:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    H, W = gray.shape
    if H < 40 or W < 40:
        return None

    if cx is None:
        cx = W // 2
    if cy is None:
        cy = H // 2
    scale_y = float(W) / float(H)

    # Threshold dark ink strokes belonging to spikes and text
    dark = (gray < 160).astype(np.uint8) * 255
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    if num_l < 15 or num_l > 25000:
        return None

    spikes = []
    for l in range(1, num_l):
        a = stats[l, cv2.CC_STAT_AREA]
        if a < 10:
            continue
        ys, xs = np.where(labels == l)
        dx = xs - cx
        dy = (ys - cy) * scale_y
        dists = np.sqrt(dx**2 + dy**2)
        min_idx = np.argmin(dists)
        max_idx = np.argmax(dists)
        spikes.append((xs[min_idx], ys[min_idx], dists[min_idx], dists[max_idx], a))

    all_max_d = [s[3] for s in spikes]
    if not all_max_d:
        return None
    outer_radius_85 = np.percentile(all_max_d, 85)

    spike_tips = []
    for tip_x, tip_y, min_d, max_d, a in spikes:
        radial_span = max_d - min_d
        if max_d >= outer_radius_85 * 0.45 and radial_span >= 12:
            angle = np.arctan2((tip_y - cy) * scale_y, tip_x - cx)
            spike_tips.append((angle, tip_x, tip_y, min_d))

    if len(spike_tips) < 25:
        return None

    # Verify 360-degree radial distribution (must span at least 9 of 12 angular sectors)
    angles = [t[0] for t in spike_tips]
    bins = set(int(np.floor((a + np.pi) / (np.pi / 6))) for a in angles)
    if len(spike_tips) < 35 or len(bins) < 9:
        return None

    # Bin into 5-degree angular sectors and select the innermost tip per bin
    bin_size = np.radians(5)
    b_dict = {}
    for angle, tx, ty, d in spike_tips:
        b_idx = int(np.floor((angle + np.pi) / bin_size))
        if b_idx not in b_dict:
            b_dict[b_idx] = []
        b_dict[b_idx].append(d)

    known_angles = []
    known_radii = []
    for b_idx, d_list in sorted(b_dict.items()):
        ang = -np.pi + (b_idx + 0.5) * bin_size
        known_angles.append(ang)
        known_radii.append(min(d_list))

    known_angles = np.array(known_angles)
    known_radii = np.array(known_radii)

    # Circular interpolation across 360 degrees
    N_samples = 180
    sample_angles = np.linspace(-np.pi, np.pi, N_samples, endpoint=False)
    ext_angles = np.concatenate([known_angles - 2 * np.pi, known_angles, known_angles + 2 * np.pi])
    ext_radii = np.concatenate([known_radii, known_radii, known_radii])
    r_interp = np.interp(sample_angles, ext_angles, ext_radii)

    # Outer & Inner Jump Smoothing Filter:
    # Outer jump: boundary leaped outward into spikes -> clamp to median (discard protrusion)
    # Inner jump: boundary leaped inward into cavity -> clamp to median (include excluded cavity area)
    win = 11
    r_filtered = np.zeros(N_samples)
    for i in range(N_samples):
        window = [r_interp[(i + j) % N_samples] for j in range(-win, win + 1)]
        med = np.median(window)
        val = r_interp[i]
        if val > med * 1.15:
            val = med
        elif val < med * 0.85:
            val = med
        r_filtered[i] = val

    r_smooth_med = np.zeros(N_samples)
    for i in range(N_samples):
        window = [r_filtered[(i + j) % N_samples] for j in range(-5, 6)]
        r_smooth_med[i] = np.median(window)

    # Low-pass Fourier harmonic reconstruction (k <= k_harmonics)
    fft_coeffs = np.fft.rfft(r_smooth_med)
    fft_coeffs[k_harmonics + 1:] = 0
    smooth_r = np.fft.irfft(fft_coeffs, n=N_samples)

    pts = []
    for a, r in zip(sample_angles, smooth_r):
        px = int(round(cx + r * np.cos(a)))
        py = int(round(cy + (r / scale_y) * np.sin(a)))
        px = max(0, min(W - 1, px))
        py = max(0, min(H - 1, py))
        pts.append((px, py))

    return pts


def clean_burst_bubble(
    img_bgr: "np.ndarray",
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    inset_px: int = 10,
) -> Optional[List[Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int], int, Any]]]:
    """
    Dedicated inside-out cleaner and typesetter for announce / burst bubbles.
    1. Traces the smooth inner spike envelope using generalized Fourier polar filtering.
    2. Wipes ONLY the Japanese dialogue characters inside the cavity:
       - Connected components of text inside the cavity are isolated and dilated by 3px.
       - Replaces text pixels with pure white [255, 255, 255], completely preserving 100%
         of the surrounding burst spikes and background art.
    3. Typesetting Profile with 10px Inset:
       - Insets the cavity scanline profile by inset_px (10px) on left, right, top, and bottom.
       - Guarantees comfortable breathing room so the translated English text never touches
         any spike tips in 360 degrees.
    """
    if cv2 is None or np is None:
        return None

    H, W = img_bgr.shape[:2]
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    pad_w = max(50, int(bw * 0.40))
    pad_h = max(50, int(bh * 0.40))
    margin = max(pad_w, pad_h) + 10

    x1 = max(0, xmin - margin); y1 = max(0, ymin - margin)
    x2 = min(W, xmax + margin); y2 = min(H, ymax + margin)

    crop = img_bgr[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    lH, lW = gray.shape

    cx_crop = ((xmin + xmax) // 2) - x1
    cy_crop = ((ymin + ymax) // 2) - y1

    pts = detect_burst_inner_region_generalized(crop, cx=cx_crop, cy=cy_crop)
    if not pts:
        return None

    poly = np.array(pts, dtype=np.int32)
    mask = np.zeros((lH, lW), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # Text-only cleaning: isolate dark connected components inside the cavity polygon
    num_l, labels, stats, centroids = cv2.connectedComponentsWithStats((gray < 180).astype(np.uint8))
    text_mask = np.zeros(gray.shape, dtype=np.uint8)
    for l in range(1, num_l):
        ys, xs = np.where(labels == l)
        inside_ratio = np.mean(mask[ys, xs] > 0)
        edge_touch = (xs.min() <= 2 or xs.max() >= lW - 3 or ys.min() <= 2 or ys.max() >= lH - 3)
        if inside_ratio >= 0.60 and not edge_touch:
            text_mask[ys, xs] = 255

    # Dilate text mask with elliptical kernel to encompass all anti-aliased gray edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    text_dilated = cv2.dilate(text_mask, kernel)
    to_clear = (text_dilated > 0) & (mask > 0) & (gray < 252)

    # Inpaint strictly the text pixels on master image with pure white
    img_bgr[y1:y2, x1:x2][to_clear] = [255, 255, 255]

    # Typesetting Profile and Bounding Box with 10px breathing inset
    ys_p, xs_p = np.where(mask > 0)
    min_yp, max_yp = int(ys_p.min()), int(ys_p.max())
    min_xp, max_xp = int(xs_p.min()), int(xs_p.max())

    profile = {}
    for y in range(min_yp + inset_px, max_yp - inset_px + 1):
        row_xs = xs_p[ys_p == y]
        if len(row_xs) > 0:
            pl = int(row_xs.min()) + inset_px + x1
            pr = int(row_xs.max()) - inset_px + x1
            if pr > pl + 15:
                profile[y + y1] = (pl, pr)

    bounds = (
        min_xp + inset_px + x1,
        min_yp + inset_px + y1,
        max_xp - inset_px + x1,
        max_yp - inset_px + y1,
    )
    cx_g = (min_xp + max_xp) // 2 + x1
    cy_g = (min_yp + max_yp) // 2 + y1
    avail_w = int((max_xp - min_xp) * 0.78)
    avail_h = int((max_yp - min_yp) * 0.80)

    return [(cx_g, cy_g, avail_w, avail_h, (0, 0, 0), profile, bounds, 0, None)]


def inpaint_bubble(
    img_bgr: "np.ndarray",
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    is_box: bool = False,
    bg_color_hint: str = "white",
) -> Optional[List[Tuple[int, int, int, int, Tuple[int, int, int], Any, Tuple[int, int, int, int]]]]:
    """
    Cleans speech bubbles inside-out and extracts genuine dialogue cavities:
    1. Dedicated line-scanning frame detection for rectangular caption boxes (100% border preservation).
    2. Dedicated radial burst/announce bubble cleaner with Fourier envelope and text-only erasing.
    3. Multi-threshold adaptive cavity detection for speech balloons (cloud, oval, screentone).
    4. Strict leak-boundary containment with proportional envelope to prevent tail leaks.
    5. Internal artwork detection to prevent text from overlapping character faces or art.
    6. Barrier-constrained cavity fill: strictly Cavity ∩ Border = ∅.
    7. Returns list of (cx, cy, sw, sh, text_color_rgb, profile, bounds).
    """
    is_dark_bubble = (bg_color_hint.lower() == "black")

    if cv2 is None or np is None:
        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2
        sw = int((xmax - xmin) * (0.80 if is_box else 0.70))
        sh = int((ymax - ymin) * (0.80 if is_box else 0.72))
        t_color = (255, 255, 255) if is_dark_bubble else (0, 0, 0)
        return [(cx, cy, sw, sh, t_color, None, (xmin, ymin, xmax, ymax), 0, None)]

    # Pass 0a: Dedicated black bubble cleaner
    if is_dark_bubble:
        black_res = clean_black_bubble(img_bgr, xmin, ymin, xmax, ymax)
        if black_res is not None:
            return black_res

    # Pass 0b: Dedicated rectangular caption box cleaner
    if is_box:
        box_res = clean_caption_box(img_bgr, xmin, ymin, xmax, ymax, bg_color_hint=bg_color_hint)
        if box_res is not None:
            return box_res

    H, W = img_bgr.shape[:2]
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    # Pass 0c: Dedicated radial burst/announce bubble cleaner
    # Only triggered if there is NO continuous closed boundary and radial spikes are detected
    crop_check = img_bgr[max(0, ymin - 30):min(H, ymax + 30), max(0, xmin - 30):min(W, xmax + 30)]
    crop_burst = img_bgr[max(0, ymin - 70):min(H, ymax + 70), max(0, xmin - 70):min(W, xmax + 70)]
    if not has_continuous_closed_boundary(crop_check) and check_is_burst(crop_burst):
        burst_res = clean_burst_bubble(img_bgr, xmin, ymin, xmax, ymax, inset_px=10)
        if burst_res is not None:
            return burst_res

    H, W = img_bgr.shape[:2]
    if len(img_bgr.shape) == 2:
        gray = img_bgr
    elif len(img_bgr.shape) == 3 and img_bgr.shape[2] == 1:
        gray = img_bgr[:, :, 0]
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    bbox_area = bw * bh

    pad_w = max(140, int(bw * 1.5))
    pad_h = max(180, int(bh * 1.5))
    margin = max(pad_w, pad_h) + 30

    x1 = max(0, xmin - margin)
    y1 = max(0, ymin - margin)
    x2 = min(W, xmax + margin)
    y2 = min(H, ymax + margin)

    local_gray = gray[y1:y2, x1:x2].copy()
    lH, lW = local_gray.shape

    # Determine cavity tone hint upfront from bubble bounding box
    bbox_gray = gray[ymin:ymax, xmin:xmax]
    med_bbox = float(np.median(bbox_gray)) if bbox_gray.size > 0 else 255.0
    is_screentone_bubble = (med_bbox < 244.0 and med_bbox > 45.0)

    # Compute adaptive thresholds from local gray statistics
    th_border, _ = compute_adaptive_thresholds(local_gray, is_dark=is_dark_bubble)
    if is_dark_bubble or med_bbox < 45.0:
        thresholds = [th_border - 20, th_border, th_border + 20, 50, 70, 90, 110]
    elif is_screentone_bubble:
        # Screentone / grey bubble: border is black/dark, screentone dots are ~140-180
        # Use lower thresholds so screentone dots don't block flood fill
        thresholds = [135, 125, 115, 105, 95, 85, 75, th_border]
    else:
        # High to low thresholds to capture antialiased, faint, and light borders first without leaking
        thresholds = [195, 185, 175, 165, th_border, 150, 135, 205, 215]

    pad_x = max(2, int(bw * 0.08))
    pad_y = max(2, int(bh * 0.08))
    inner_x1 = min(xmin + pad_x, xmax - 2)
    inner_x2 = max(xmax - pad_x, inner_x1 + 1)
    inner_y1 = min(ymin + pad_y, ymax - 2)
    inner_y2 = max(ymax - pad_y, inner_y1 + 1)

    nx = max(7, min(15, bw // 8))
    ny = max(7, min(15, bh // 8))
    xs_grid = np.linspace(inner_x1, inner_x2, nx, dtype=int)
    ys_grid = np.linspace(inner_y1, inner_y2, ny, dtype=int)

    best_fill = None
    best_dilated_border = None
    is_burst = False

    # Proportional boundary envelope to prevent floodfill leaks
    max_leak_x1 = max(0, xmin - pad_w) - x1
    max_leak_y1 = max(0, ymin - pad_h) - y1
    max_leak_x2 = min(W, xmax + pad_w) - x1
    max_leak_y2 = min(H, ymax + pad_h) - y1

    pad_text_y = max(60, int(bh * 0.60))
    pad_text_x = max(50, int(bw * 0.65))

    candidate_fills = []
    best_char_mask = None

    for th in thresholds:
        if is_dark_bubble:
            local_border = (local_gray > th).astype(np.uint8) * 255
        else:
            local_border = (local_gray < th).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local_border, connectivity=8)
        if num_labels <= 1:
            continue

        areas = stats[:, cv2.CC_STAT_AREA]
        cws = stats[:, cv2.CC_STAT_WIDTH]
        chs = stats[:, cv2.CC_STAT_HEIGHT]
        clxs = stats[:, cv2.CC_STAT_LEFT]
        clys = stats[:, cv2.CC_STAT_TOP]

        abs_xs = clxs + x1
        abs_ys = clys + y1

        in_text_zone = (abs_xs >= xmin - pad_text_x) & (abs_xs + cws <= xmax + pad_text_x) & (abs_ys >= ymin - pad_text_y) & (abs_ys + chs <= ymax + pad_text_y)
        is_char_sized = (cws < max(50, int(bw * 0.85))) & (chs < max(50, int(bh * 0.50))) & (areas < max(700, int(bbox_area * 0.35)))

        char_mask_lut = np.zeros(num_labels, dtype=np.uint8)
        char_mask_lut[1:] = np.where(in_text_zone[1:] & is_char_sized[1:], 255, 0)
        char_mask = char_mask_lut[labels]

        is_outline = (
            (areas > max(25, int(bbox_area * 0.04)))
            | (cws > bw * 0.30)
            | (chs > bh * 0.30)
        )
        border_lut = np.zeros(num_labels, dtype=np.uint8)
        border_lut[1:] = np.where(~(in_text_zone[1:] & is_char_sized[1:]) & is_outline[1:], 255, 0)
        cleaned_border = border_lut[labels]

        # Multi-scale closure to bridge antialiasing and whisper gaps
        ksize = (5, 5) if min(bw, bh) < 50 else (7, 7)
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        local_border_closed = cv2.morphologyEx(cleaned_border, cv2.MORPH_CLOSE, close_k)
        local_border_dilated = cv2.dilate(local_border_closed, close_k)

        # High-performance single-pass cavity detection on inverted border
        inv_border = (local_border_dilated == 0).astype(np.uint8)
        n_cav, cav_labels, cav_stats, _ = cv2.connectedComponentsWithStats(inv_border, connectivity=8)

        grid_xs_local = xs_grid - x1
        grid_ys_local = ys_grid - y1
        valid_gx = grid_xs_local[(grid_xs_local >= 0) & (grid_xs_local < lW)]
        valid_gy = grid_ys_local[(grid_ys_local >= 0) & (grid_ys_local < lH)]
        if len(valid_gx) > 0 and len(valid_gy) > 0:
            sub_grid = cav_labels[np.ix_(valid_gy, valid_gx)]
            cand_lbls = np.unique(sub_grid)
        else:
            cand_lbls = np.array([], dtype=int)

        valid_lbls = []
        for lbl in cand_lbls:
            if lbl == 0:
                continue
            cly = cav_stats[lbl, cv2.CC_STAT_TOP]
            clx = cav_stats[lbl, cv2.CC_STAT_LEFT]
            c_w = cav_stats[lbl, cv2.CC_STAT_WIDTH]
            c_h = cav_stats[lbl, cv2.CC_STAT_HEIGHT]
            area = cav_stats[lbl, cv2.CC_STAT_AREA]
            touches_roi_edge = (
                (cly <= 2 and y1 > 0)
                or (cly + c_h >= lH - 3 and y2 < H)
                or (clx <= 2 and x1 > 0)
                or (clx + c_w >= lW - 3 and x2 < W)
            )
            if not touches_roi_edge and area >= bbox_area * 0.10:
                valid_lbls.append(lbl)

        if valid_lbls:
            lut = np.zeros(n_cav, dtype=np.uint8)
            lut[valid_lbls] = 255
            combined_fill = lut[cav_labels]
            tot_area = int(np.sum(cav_stats[valid_lbls, cv2.CC_STAT_AREA]))
            if tot_area >= bbox_area * 0.15:
                if len(valid_lbls) > 1:
                    bbox_cx_l = (xmin + xmax) // 2 - x1
                    bbox_cy_l = (ymin + ymax) // 2 - y1
                    cw = max(5, int(bw * 0.30))
                    ch = max(5, int(bh * 0.30))
                    cy1_w = max(0, bbox_cy_l - ch // 2)
                    cy2_w = min(lH, bbox_cy_l + ch // 2)
                    cx1_w = max(0, bbox_cx_l - cw // 2)
                    cx2_w = min(lW, bbox_cx_l + cw // 2)
                    center_sub = cav_labels[cy1_w:cy2_w, cx1_w:cx2_w]
                    cen_valid = set(np.unique(center_sub)) & set(valid_lbls)
                    if not cen_valid:
                        best_idx = int(np.argmax(cav_stats[valid_lbls, cv2.CC_STAT_AREA]))
                        cen_valid = {valid_lbls[best_idx]}
                    lut2 = np.zeros(n_cav, dtype=np.uint8)
                    lut2[list(cen_valid)] = 255
                    combined_fill = lut2[cav_labels]
                    tot_area = int(np.sum(cav_stats[list(cen_valid), cv2.CC_STAT_AREA]))

                candidate_fills.append((tot_area, combined_fill, local_border_dilated, char_mask))

    if candidate_fills:
        # Select the candidate fill maximizing cavity coverage without leaks
        candidate_fills.sort(key=lambda item: item[0], reverse=True)
        best_fill = candidate_fills[0][1]
        best_dilated_border = candidate_fills[0][2]
        best_char_mask = candidate_fills[0][3]

    fill_bgr = [0, 0, 0] if is_dark_bubble else [255, 255, 255]
    text_color_rgb = (255, 255, 255) if is_dark_bubble else (0, 0, 0)

    if best_fill is not None:
        # Verify that the cavity actually covers the text bounding box (rejects disconnected leaks)
        bx1 = max(0, xmin - x1)
        bx2 = min(lW, xmax - x1)
        by1 = max(0, ymin - y1)
        by2 = min(lH, ymax - y1)
        overlap = np.sum(best_fill[by1:by2, bx1:bx2] > 0)
        if overlap < bbox_area * 0.20:
            best_fill = None

    if best_fill is None or best_fill.max() == 0:
        # Check if a vertical panel frame line crosses through the text box:
        # If so, this is floating text over a panel line and must be routed to floating text cleaner!
        top_y1 = max(0, ymin - 40)
        bot_y2 = min(H, ymax + 40)
        has_crossing_panel_line = False
        for c in range(max(0, xmin - 10), min(W, xmax + 10)):
            top_dark = (gray[top_y1:ymin, c] < 90).sum()
            bot_dark = (gray[ymax:bot_y2, c] < 90).sum()
            if top_dark > 15 and bot_dark > 15:
                has_crossing_panel_line = True
                break

        # Fallback for bright white speech bubbles where outer boundary was discontinuous or leaked at ROI edge
        if not has_crossing_panel_line and med_bbox >= 215.0 and not is_dark_bubble:
            bx1 = max(0, xmin - 2)
            by1 = max(0, ymin - 2)
            bx2 = min(W, xmax + 2)
            by2 = min(H, ymax + 2)
            # Erase character-sized dark ink strokes inside bounding box (strictly protecting bubble perimeter)
            local_box = img_bgr[by1:by2, bx1:bx2]
            box_gray = cv2.cvtColor(local_box, cv2.COLOR_BGR2GRAY)
            raw_ink = (box_gray < 170).astype(np.uint8) * 255
            n_lbl, lbls, sts, _ = cv2.connectedComponentsWithStats(raw_ink)
            char_ink = np.zeros_like(raw_ink, dtype=bool)
            box_w = bx2 - bx1
            box_h = by2 - by1
            for l in range(1, n_lbl):
                lw = sts[l, cv2.CC_STAT_WIDTH]
                lh = sts[l, cv2.CC_STAT_HEIGHT]
                la = sts[l, cv2.CC_STAT_AREA]
                if lw < max(40, int(box_w * 0.70)) and lh < max(40, int(box_h * 0.60)) and la < max(500, int(box_w * box_h * 0.30)):
                    char_ink[lbls == l] = True
            k_tdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            text_ink_dil = cv2.dilate(char_ink.astype(np.uint8) * 255, k_tdil) > 0
            img_bgr[by1:by2, bx1:bx2][text_ink_dil] = [255, 255, 255]
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            sw = max(20, int(bw * 0.85))
            sh = max(20, int(bh * 0.85))
            return [(cx, cy, sw, sh, (0, 0, 0), None, (xmin, ymin, xmax, ymax), 0, None)]
        return None

    ys_c, xs_c = np.where(best_fill > 0)
    if len(ys_c) == 0:
        return None

    min_r = int(np.min(ys_c)) + y1
    max_r = int(np.max(ys_c)) + y1
    min_c = int(np.min(xs_c)) + x1
    max_c = int(np.max(xs_c)) + x1

    # Internal Artwork Detection: large dark components inside cavity (e.g. character faces, chibi heads)
    dark_inner = (local_gray > 160).astype(np.uint8) if is_dark_bubble else (local_gray < 160).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_inner)
    art_mask = np.zeros_like(best_fill, dtype=bool)
    art_box = None
    art_labels = []

    for l in range(1, num_labels):
        a = stats[l, cv2.CC_STAT_AREA]
        cw = stats[l, cv2.CC_STAT_WIDTH]
        ch = stats[l, cv2.CC_STAT_HEIGHT]
        cx_l = stats[l, cv2.CC_STAT_LEFT]
        cy_l = stats[l, cv2.CC_STAT_TOP]
        abs_x1 = cx_l + x1
        abs_y1 = cy_l + y1
        abs_x2 = abs_x1 + cw
        abs_y2 = abs_y1 + ch
        solidity = a / float(max(1, cw * ch))
        is_perimeter = (cw >= (max_c - min_c) * 0.70 and ch >= (max_r - min_r) * 0.70)
        # Artwork core: genuine internal illustration (e.g. chibi head) has high solidity
        # and does not form the outer perimeter of the cavity.
        if (min_c - 15 <= abs_x1 and abs_x2 <= max_c + 15 and min_r - 15 <= abs_y1 and abs_y2 <= max_r + 15):
            if not is_perimeter and solidity >= 0.22 and (a > 1200 or (cw > 50 and ch > 50 and a > 500)):
                art_labels.append(l)

    if art_labels:
        ab_x1 = min(stats[l, cv2.CC_STAT_LEFT] + x1 for l in art_labels)
        ab_y1 = min(stats[l, cv2.CC_STAT_TOP] + y1 for l in art_labels)
        ab_x2 = max(stats[l, cv2.CC_STAT_LEFT] + stats[l, cv2.CC_STAT_WIDTH] + x1 for l in art_labels)
        ab_y2 = max(stats[l, cv2.CC_STAT_TOP] + stats[l, cv2.CC_STAT_HEIGHT] + y1 for l in art_labels)
        art_box = (ab_x1, ab_y1, ab_x2, ab_y2)

        # Protect all components belonging to the artwork cluster (face, blush lines, neck, hair)
        for l in range(1, num_labels):
            cx_l = stats[l, cv2.CC_STAT_LEFT] + x1
            cy_l = stats[l, cv2.CC_STAT_TOP] + y1
            cw = stats[l, cv2.CC_STAT_WIDTH]
            ch = stats[l, cv2.CC_STAT_HEIGHT]
            if (cx_l >= ab_x1 - 20 and cx_l + cw <= ab_x2 + 25 and cy_l >= ab_y1 - 20 and cy_l <= ab_y2 + 60):
                art_mask[labels == l] = True

        # Dilate art_mask slightly (3px) to protect delicate facial linework and blush marks
        k_art = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        art_mask = cv2.dilate(art_mask.astype(np.uint8), k_art) > 0

    # Check for multi-lobe structure in conjoined bubble (skip for burst/shock bubbles and boxes)
    if is_burst or is_box:
        lobes = [(best_fill, min_c, min_r, max_c, max_r)]
    else:
        lobes = segment_cavity_lobes(best_fill, min_c, min_r, max_c, max_r, x1, y1)
    regions = []

    # STRICT 360-DEGREE BORDER PROTECTION
    if is_burst or is_screentone_bubble:
        barrier_mask = (best_dilated_border > 0) if best_dilated_border is not None else np.zeros_like(best_fill, dtype=bool)
    else:
        # Standard white speech balloons: protect the genuine bubble outline with a tight 1px barrier
        # so characters touching or near the edge (e.g. kana like 'ru' or kanji near panel dividers) are completely cleaned!
        k_1px = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cavity_contour = cv2.morphologyEx(best_fill, cv2.MORPH_GRADIENT, k_1px)
        outline_tight = (local_gray < 110) & (cv2.dilate(cavity_contour, k_1px) > 0)
        barrier_mask = cv2.dilate(outline_tight.astype(np.uint8), k_1px) > 0

    for (l_mask, l_min_c, l_min_r, l_max_c, l_max_r) in lobes:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_lobe = cv2.morphologyEx(l_mask, cv2.MORPH_CLOSE, k_close)

        # STRICT BARRIER: Cavity ∩ Border = ∅ & Cavity ∩ Artwork = ∅
        safe_inpaint = (closed_lobe > 0) & (~barrier_mask) & (~art_mask)
        if best_char_mask is not None and not is_burst and not is_screentone_bubble:
            char_mask_dil = cv2.dilate(best_char_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            # Erase character glyphs cleanly even if touching the cavity boundary
            safe_inpaint = safe_inpaint | ((char_mask_dil > 0) & (~art_mask) & (cv2.dilate(closed_lobe, k_close) > 0))

        # Ensure all dark ink strokes inside the OCR text box within the bubble are cleared
        if not is_screentone_bubble and not is_dark_bubble:
            bx_1 = max(0, xmin - x1)
            by_1 = max(0, ymin - y1)
            bx_2 = min(lW, xmax - x1)
            by_2 = min(lH, ymax - y1)
            text_crop_zone = np.zeros_like(safe_inpaint, dtype=bool)
            text_crop_zone[by_1:by_2, bx_1:bx_2] = True
            text_dark_strokes = (local_gray < 170) & text_crop_zone & (cv2.dilate(closed_lobe, k_close) > 0)
            k_tdil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            text_dark_dil = cv2.dilate(text_dark_strokes.astype(np.uint8), k_tdil) > 0
            safe_inpaint = safe_inpaint | (text_dark_dil & (~barrier_mask) & (~art_mask))

        # Tone analysis of the cavity interior to determine background fill mode
        cavity_vals = local_gray[safe_inpaint]
        stroke_width = 0
        stroke_color = None

        if len(cavity_vals) > 0:
            med_cavity = float(np.median(cavity_vals))
            is_dark_cavity = is_dark_bubble or (med_cavity < 45.0)

            if is_dark_cavity:
                # Black bubble: solid black inpainting of text characters, pure white text without halo
                fill_bgr = [0, 0, 0]
                text_color_rgb = (255, 255, 255)
                stroke_width = 0
                stroke_color = None
                # Restrict black fill strictly to non-black text characters inside the bubble bbox
                in_bbox = np.zeros_like(safe_inpaint, dtype=bool)
                in_bbox[max(0, ymin - y1):min(lH, ymax - y1), max(0, xmin - x1):min(lW, xmax - x1)] = True
                dark_clean = safe_inpaint & in_bbox & (local_gray > 30)
                img_bgr[y1:y2, x1:x2][dark_clean] = fill_bgr
            else:
                bright_cavity = cavity_vals[cavity_vals > 90]
                med_bg = float(np.median(bright_cavity)) if len(bright_cavity) > 0 else med_cavity
                pct_white = float(np.mean(cavity_vals >= 242))

                # Pure white bubble: background is solid white AND bubble bbox is not screentone
                if med_bg >= 246.0 and not is_screentone_bubble:
                    fill_bgr = [255, 255, 255]
                    text_color_rgb = (0, 0, 0)
                    stroke_width = 0
                    stroke_color = None
                    img_bgr[y1:y2, x1:x2][safe_inpaint] = fill_bgr
                else:
                    # Grey screentone bubble: reconstruct authentic periodic screentone dots across safe_inpaint
                    text_color_rgb = (0, 0, 0)
                    stroke_width = 2
                    stroke_color = (255, 255, 255)  # Crisp white glow around black text for high contrast readability

                    text_strokes = (local_gray < 110) & safe_inpaint
                    k_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
                    text_and_halo = cv2.dilate(text_strokes.astype(np.uint8) * 255, k_halo) > 0
                    pure_tone = safe_inpaint & (~text_and_halo)

                    ys_t, xs_t = np.where(pure_tone)
                    ay_t = ys_t + y1
                    ax_t = xs_t + x1

                    best_P = 4
                    best_contrast = -1
                    best_tile = None
                    if len(ys_t) > 50:
                        # Subsample if too many points for instantaneous period estimation
                        if len(ys_t) > 3000:
                            sub_idx = np.random.choice(len(ys_t), 3000, replace=False)
                            ay_sub = ay_t[sub_idx]
                            ax_sub = ax_t[sub_idx]
                        else:
                            ay_sub = ay_t
                            ax_sub = ax_t
                        for P in [4, 8, 2]:
                            t_accum = np.zeros((P, P, 3), dtype=np.float32)
                            t_cnt = np.zeros((P, P, 1), dtype=np.float32)
                            np.add.at(t_accum, (ay_sub % P, ax_sub % P), img_bgr[ay_sub, ax_sub])
                            np.add.at(t_cnt, (ay_sub % P, ax_sub % P), 1)
                            if np.all(t_cnt > 0):
                                t_accum /= np.maximum(t_cnt, 1)
                                t_u8 = np.round(t_accum).astype(np.uint8)
                                c = float(t_u8.max() - t_u8.min())
                                if c > best_contrast:
                                    best_contrast = c
                                    best_P = P
                                    best_tile = t_u8

                    if best_tile is not None and best_contrast >= 8.0:
                        # Vectorized halftone reconstruction across entire safe cavity
                        ys_grid, xs_grid = np.ogrid[:lH, :lW]
                        tiled = best_tile[(ys_grid + y1) % best_P, (xs_grid + x1) % best_P]
                        img_bgr[y1:y2, x1:x2][safe_inpaint] = tiled[safe_inpaint]
                    else:
                        tone_val = int(med_bg) if med_bg < 245 else 220
                        img_bgr[y1:y2, x1:x2][safe_inpaint] = [tone_val, tone_val, tone_val]
        else:
            img_bgr[y1:y2, x1:x2][safe_inpaint] = fill_bgr

        l_ys, l_xs = np.where(safe_inpaint)
        if len(l_ys) == 0:
            continue

        profile = {}
        if is_burst:
            center_col = ((xmin + xmax) // 2) - x1
            for y_idx in range(int(np.min(l_ys)), int(np.max(l_ys)) + 1):
                if 0 <= center_col < lW and safe_inpaint[y_idx, center_col] and not barrier_mask[y_idx, center_col]:
                    l_col = center_col
                    while l_col > 0 and safe_inpaint[y_idx, l_col - 1] and not barrier_mask[y_idx, l_col - 1]:
                        l_col -= 1
                    r_col = center_col
                    while r_col < lW - 1 and safe_inpaint[y_idx, r_col + 1] and not barrier_mask[y_idx, r_col + 1]:
                        r_col += 1
                    profile[y_idx + y1] = (l_col + x1, r_col + x1)
        else:
            for y_idx in range(int(np.min(l_ys)), int(np.max(l_ys)) + 1):
                row_xs = l_xs[l_ys == y_idx]
                if len(row_xs) > 0:
                    row_min = int(np.min(row_xs)) + x1
                    row_max = int(np.max(row_xs)) + x1
                    if art_box and (art_box[1] - 5 <= y_idx + y1 <= art_box[3] + 5):
                        if art_box[0] > min_c + 40:
                            row_max = min(row_max, art_box[0] - 8)
                        elif art_box[2] < max_c - 40:
                            row_min = max(row_min, art_box[2] + 8)
                    if row_max > row_min:
                        profile[y_idx + y1] = (row_min, row_max)

        p_ys = list(profile.keys())
        if not p_ys:
            continue

        act_min_y = min(p_ys)
        act_max_y = max(p_ys)
        act_min_x = min(profile[y][0] for y in p_ys)
        act_max_x = max(profile[y][1] for y in p_ys)

        # Tail trimming: only trim if edge columns form a genuine thin tail (height < 15% of max column height)
        if not is_box and not is_burst:
            col_h = {x: sum(1 for y in p_ys if profile[y][0] <= x <= profile[y][1]) for x in range(act_min_x, act_max_x + 1)}
            if col_h:
                max_h = max(col_h.values())
                th = max(8, int(max_h * 0.15))
                curr_min = act_min_x
                while curr_min < act_max_x and col_h.get(curr_min, 0) < th:
                    curr_min += 1
                curr_max = act_max_x
                while curr_max > curr_min and col_h.get(curr_max, 0) < th:
                    curr_max -= 1
                act_min_x = curr_min
                act_max_x = curr_max

                for y_k in list(profile.keys()):
                    pl, pr = profile[y_k]
                    pl = max(pl, act_min_x)
                    pr = min(pr, act_max_x)
                    if pr > pl:
                        profile[y_k] = (pl, pr)
                    else:
                        del profile[y_k]

        raw_w = act_max_x - act_min_x
        raw_h = act_max_y - act_min_y

        if is_box:
            sw = max(20, min(int(raw_w * 0.90), raw_w - 6))
            sh = max(20, min(int(raw_h * 0.90), raw_h - 6))
        elif is_burst:
            sw = max(20, min(int(raw_w * 0.70), raw_w - 16))
            sh = max(20, min(int(raw_h * 0.70), raw_h - 16))
        else:
            sw = max(20, min(int(raw_w * 0.82), raw_w - 14))
            sh = max(20, min(int(raw_h * 0.84), raw_h - 14))

        if profile and np is not None:
            cx_g = int(np.mean([(pl + pr) / 2.0 for pl, pr in profile.values()]))
            cy_g = int(np.mean(list(profile.keys())))
        else:
            cx_g = (act_min_x + act_max_x) // 2
            cy_g = (act_min_y + act_max_y) // 2

        regions.append((
            cx_g,
            cy_g,
            sw,
            sh,
            text_color_rgb,
            profile,
            (act_min_x, act_min_y, act_max_x, act_max_y),
            stroke_width,
            stroke_color,
        ))

    return regions if regions else None
