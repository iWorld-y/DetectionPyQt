"""
@Time    : 2023/6/11 4:06
@Author  : Eugene
@FileName: GetDetector.py 
"""
import os
import onnxruntime
from .DetectV5 import DetectV5
from .DetectV6 import DetectV6
from .DetectV8 import DetectV8


class GetDetector:
    def __init__(self, CLASSES, ONNX_path):
        self.CLASSES = CLASSES
        self.ONNX_path = ONNX_path

    def get_detectors(self):
        detectors = list()
        for detector, num in zip([DetectV5, DetectV6, DetectV8], [5, 6, 8]):
            detectors.append(
                detector(onnxruntime.InferenceSession(os.path.join(self.ONNX_path, f"v{num}.onnx"),
                                                      providers=['CPUExecutionProvider']), classes=self.CLASSES))
        return detectors
