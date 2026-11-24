"""Computer vision, cavity extraction, border protection, and inpainting modules."""

from .analyzer import analyze_region_tone, compute_adaptive_thresholds
from .bubble_cleaner import inpaint_bubble
from .conjoined import is_lobe_viable_for_text, partition_overlapping_regions, segment_cavity_lobes
from .floating_cleaner import inpaint_floating

__all__ = [
    "analyze_region_tone",
    "compute_adaptive_thresholds",
    "inpaint_bubble",
    "segment_cavity_lobes",
    "is_lobe_viable_for_text",
    "partition_overlapping_regions",
    "inpaint_floating",
]
