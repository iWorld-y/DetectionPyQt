"""
@Time    : 2023/6/11 4:27
@Author  : Eugene
@FileName: test.py 
"""
from src.YOLO_ONNX_Detection.GetDetector import GetDetector

get_detector = GetDetector(
    ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
     "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
     "sling dress"], "../models").get_detectors()
print(get_detector[0].classes)
print(get_detector[1])
print(get_detector[2])
