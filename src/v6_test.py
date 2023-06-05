# %%
import os, sys
import cv2
import matplotlib.pyplot as plt

sys.path.append("/home/eugene/code/YOLOv6")
from yolov6.core.inferer import Inferer

CLASSES = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
           "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
           "sling dress"]  # coco80类别

img_path = os.path.join("/home/eugene/autodl-tmp/test", "1.jpg")
weight_path = "/home/eugene/autodl-tmp/weights/v6.pt"
device = "0"
yaml_path = "/home/eugene/code/Multi-labelClothingDetection/dataset/coco.yaml"
# inferer = Inferer(weights=weight_path, source=img_path, device=device)
inferer = Inferer(source=img_path, webcam=False, webcam_addr=0, weights=weight_path, device=device, yaml=yaml_path,
                  img_size=640, half=False)
res_img = inferer.get_infered_img(conf_thres=0.4, iou_thres=0.4, classes=None, agnostic_nms=False, max_det=1000)
plt.imshow(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
plt.show()
