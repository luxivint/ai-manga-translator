import numpy as np

from ..utils.textblock import TextBlock
from .factory import DetectionEngineFactory


class TextBlockDetector:
    """
    Detector for finding text blocks in images.
    """
    
    def __init__(self, settings_page):
        self.settings = settings_page 
        self.detector = 'RT-DETR-v2'  # Default Detector
    
    def detect(self, img: np.ndarray) -> list[TextBlock]:
        self.detector = self.settings.get_tool_selection('detector') or self.detector
        engine = DetectionEngineFactory.create_engine(
            self.settings, self.detector
        )
        return engine.detect(img)

    def _engine(self):
        self.detector = self.settings.get_tool_selection('detector') or self.detector
        return DetectionEngineFactory.create_engine(self.settings,self.detector)

    def reuse_identity(self):
        return getattr(self._engine(),'reuse_identity',None)

    def detect_from_regions(self,image,text_boxes,bubble_boxes):
        """Same TextBlock/font/colour contract without a second region detection."""
        return self._engine().create_text_blocks(image,text_boxes,bubble_boxes)
