import cv2
import numpy as np

from .base import InpaintModel
from .schema import Config


class SmartCV2(InpaintModel):
    """Fast cv2 INPAINT_TELEA with full bounding-box masking.

    Bypasses HD strategies and character-level masking entirely — cv2 handles
    any image size natively and fills from surrounding context, so full-bbox
    masks never produce white-blob artifacts on dark backgrounds.
    """

    name = "SmartCV2"

    def init_model(self, device, **kwargs):
        pass

    @staticmethod
    def is_downloaded() -> bool:
        return True

    def forward(self, image, mask, config: Config):
        raise NotImplementedError("SmartCV2 overrides __call__ directly")

    def __call__(self, image: np.ndarray, mask: np.ndarray, config: Config) -> np.ndarray:
        """
        image: [H, W, C] RGB uint8
        mask:  [H, W] uint8, 255 = region to inpaint
        returns: [H, W, C] RGB uint8
        """
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask > 127).astype(np.uint8) * 255

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, mask, inpaintRadius=15, flags=cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
