# %%
import sys
import logging
import os
import numpy as np

import onnxruntime

# 科学模式下才需此行
sys.path.append("/home/eugene/code/DetectionPyQt/src")
from DetectV5 import DetectV5
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)
# %%
img_path = r'/home/eugene/autodl-tmp/test/1.jpg'
onnx_path = r"models/v5.onnx"

CLASSES = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
           "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
           "sling dress"]
detect = DetectV5(onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider']), classes=CLASSES)

# %%
images_path = "/home/eugene/autodl-tmp/test"
for img in os.listdir(images_path):
    if (os.path.splitext(img)[-1] == ".jpg"):
        img_pred = detect.inference(cv2.imread(os.path.join(images_path, img)), conf_thres=0.5, iou_thres=0.5)
        plt.imshow(img_pred)
        logging.info(detect.result)
        logging.info(detect.score)
        plt.show()
