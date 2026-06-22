from modules.utils.device import get_providers
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session
from .rtdetr_v2_onnx import RTDetrV2ONNXDetection


class RTDetrV4SInt8ONNXDetection(RTDetrV2ONNXDetection):
    """RT-DETR v4-s, int8-quantized ONNX backend.

    Same inference pipeline as RTDetrV2ONNXDetection (identical input/output
    contract), backed by a smaller, faster int8-quantized model file.
    """

    def initialize(
        self,
        device: str = 'cpu',
        confidence_threshold: float = 0.3,
    ) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold

        file_path = ModelDownloader.get_file_path(
            ModelID.RTDETR_V4S_INT8_ONNX, 'detector-v4-s_int8.onnx'
        )
        providers = get_providers(self.device)
        self.session = make_session(file_path, providers=providers)
