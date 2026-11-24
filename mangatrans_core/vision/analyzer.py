"""Image tone, screentone detection, and adaptive edge profiling."""

from typing import Dict, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None


def analyze_region_tone(roi_gray: "np.ndarray") -> Dict[str, float]:
    """
    Computes statistical tone features of an image region:
    - mean brightness
    - median brightness
    - standard deviation
    - high-frequency screentone score
    - background polarity (is_dark)
    """
    if roi_gray is None or roi_gray.size == 0 or np is None:
        return {
            "mean": 255.0,
            "median": 255.0,
            "std": 0.0,
            "screentone_score": 0.0,
            "is_dark": False,
        }

    mean_val = float(np.mean(roi_gray))
    median_val = float(np.median(roi_gray))
    std_val = float(np.std(roi_gray))

    # High frequency variation check (Laplacian variance) for screentone/halftone pattern
    screentone_score = 0.0
    if cv2 is not None and roi_gray.shape[0] >= 8 and roi_gray.shape[1] >= 8:
        lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
        screentone_score = float(np.var(lap))

    is_dark = (median_val < 100.0)

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "screentone_score": screentone_score,
        "is_dark": is_dark,
    }


def compute_adaptive_thresholds(
    roi_gray: "np.ndarray",
    is_dark: bool = False,
) -> Tuple[int, int]:
    """
    Calculates dynamic lower and upper intensity thresholds for bubble border detection,
    avoiding static hardcoded thresholds like [175, 155, 135].
    """
    if roi_gray is None or roi_gray.size == 0 or np is None:
        return (50, 150) if is_dark else (120, 200)

    # Use percentile distribution to adapt to scanned lightness
    p15 = float(np.percentile(roi_gray, 15))
    p50 = float(np.percentile(roi_gray, 50))
    p85 = float(np.percentile(roi_gray, 85))

    if is_dark:
        th_border = int(min(220, max(60, p85 * 0.75)))
        th_cavity = int(max(20, min(140, p50)))
    else:
        th_border = int(max(30, min(180, p15 * 1.35)))
        th_cavity = int(min(250, max(180, p50)))

    return th_border, th_cavity
