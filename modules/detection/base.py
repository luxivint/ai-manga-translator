
from modules.utils import stage_trace as trace
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ..utils.textblock import TextBlock
from .utils.geometry import does_rectangle_fit, do_rectangles_overlap, \
    merge_overlapping_boxes
from .font.engine import FontEngineFactory
from .utils.content import filter_and_fix_bboxes


class DetectionEngine(ABC):
    """
    Abstract base class for all detection engines.
    Each model implementation should inherit from this class.
    """
    
    def __init__(self, settings=None):
        self.settings = settings
    
    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the detection model with necessary parameters.
        
        Args:
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[TextBlock]:
        """
        Detect text blocks in an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of TextBlock objects with detected regions
        """
        pass
        
    @trace.traced("detection.build_blocks")
    def create_text_blocks(
        self, 
        image: np.ndarray, 
        text_boxes: np.ndarray,
        bubble_boxes: Optional[np.ndarray] = None
    ) -> list[TextBlock]:
        
        text_boxes = filter_and_fix_bboxes(text_boxes, image.shape)
        bubble_boxes = filter_and_fix_bboxes(bubble_boxes, image.shape)
        # Keep the existing duplicate-box cleanup, then recover text that spans
        # two distinct touching balloons BEFORE font inference and OCR.
        from .utils.balloon_regions import assign_balloon_regions
        text_boxes = merge_overlapping_boxes(text_boxes)
        regions = assign_balloon_regions(image, text_boxes, bubble_boxes)
        text_boxes = [text_box for text_box, _, _ in regions]

        text_blocks = []
        text_matched = [False] * len(text_boxes)  # Track matched text boxes
        
        # Set bubble_boxes to empty array if None
        if bubble_boxes is None:
            bubble_boxes = np.array([])

        if len(text_boxes) == 0:
            return text_blocks

        # Batch font inference: collect all valid crops first
        h, w = image.shape[:2]
        crops = []
        crop_indices = []  # maps position in crops[] -> index in text_boxes
        for txt_idx, txt_box in enumerate(text_boxes):
            x1, y1, x2, y2 = map(int, txt_box)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
                crop_indices.append(txt_idx)

        font_attrs_per_box: list[dict] = [{}] * len(text_boxes)
        if crops:
            try:
                with trace.span("font.ensure"):
                    font_engine = FontEngineFactory.create_engine(self.settings)
                with trace.span("font.infer"):
                    batch_results = font_engine.process_batch(crops)
                for crop_pos, txt_idx in enumerate(crop_indices):
                    font_attrs_per_box[txt_idx] = batch_results[crop_pos]
            except Exception as e:
                print(f"Failed to batch detect font attributes: {e}")

        # Build TextBlock objects using pre-computed font attrs
        for txt_idx, txt_box in enumerate(text_boxes):
            font_attrs = font_attrs_per_box[txt_idx]
            direction = font_attrs.get('direction', '')
            text_color = tuple(font_attrs.get('text_color', ()))

            # The owner is chosen by overlap area (not detector order). Splits
            # retain their individual balloon identities through rendering.
            _, matched_bubble, split_from_double_balloon = regions[txt_idx]
            block = TextBlock(
                text_bbox=txt_box,
                bubble_bbox=matched_bubble,
                text_class='text_bubble' if matched_bubble is not None else 'text_free',
                direction=direction,
                font_color=text_color,
            )
            block.split_from_double_balloon = split_from_double_balloon
            text_blocks.append(block)

        return text_blocks
    
