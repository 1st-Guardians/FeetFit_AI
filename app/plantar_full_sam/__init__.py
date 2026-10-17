"""Self-contained, local-only runtime for plantar Full SAM inference.

This package contains only deterministic model and mask operations.  It has
no OpenAI client, HTTP client, or external review dependency.
"""

from .full_sam_cleanup import FullSamCleanupConfig, cleanup_full_sam_mask
from .full_sam_smoothing import FullSamSmoothingConfig, smooth_full_sam_mask
from .sole_alpha_cleanup import (
    SoleAlphaCleanupConfig,
    SoleAlphaCleanupResult,
    cleanup_sole_alpha,
    top_plateau_ratio,
)
from .sole_alpha_refiner import (
    SoleAlphaRefinementConfig,
    SoleAlphaRefinementResult,
    refine_sole_alpha_with_sam,
)

__all__ = [
    "FullSamCleanupConfig",
    "FullSamSmoothingConfig",
    "SoleAlphaCleanupConfig",
    "SoleAlphaCleanupResult",
    "SoleAlphaRefinementConfig",
    "SoleAlphaRefinementResult",
    "cleanup_full_sam_mask",
    "cleanup_sole_alpha",
    "smooth_full_sam_mask",
    "top_plateau_ratio",
    "refine_sole_alpha_with_sam",
]
