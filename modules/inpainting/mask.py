import numpy as np


def constrain_mask_to_allowed_regions(
    mask: np.ndarray,
    allowed_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Clip inpaint pixels to text regions and remove pixels marked protected."""
    if mask.ndim != 2 or allowed_mask.ndim != 2 or mask.shape != allowed_mask.shape:
        raise ValueError("mask and allowed_mask must be 2D arrays with matching shapes")

    result = (mask > 0) & (allowed_mask > 0)
    if protected_mask is not None:
        if protected_mask.ndim != 2 or protected_mask.shape != mask.shape:
            raise ValueError("protected_mask must match the mask shape")
        result &= protected_mask == 0

    return result.astype(np.uint8) * 255
