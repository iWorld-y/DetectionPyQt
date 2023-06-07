# %%
import logging

import cv2
import time
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict, namedtuple

weight = "models/v6_4.onnx"
imgList = [cv2.imread('/home/eugene/autodl-tmp/test/3.jpg')]

providers = ['CPUExecutionProvider']
session = ort.InferenceSession(weight, providers=providers)
names = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
         "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
         "sling dress"]
colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Preprocess image. For details, see:
    https://github.com/meituan/YOLOv6/issues/613
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


origin_RGB = []
resize_data = []
for img in imgList:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    origin_RGB.append(img)
    image = img.copy()
    image, ratio, dwdh = letterbox(image)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    resize_data.append((im, ratio, dwdh))
np_batch = np.concatenate([data[0] for data in resize_data])
outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]
# batch 1 infer
im = np.ascontiguousarray(np_batch[0:1, :] / 255)
out = session.run(outname, {'images': im})
# %%
for i in range(out[0].shape[0]):
    obj_num = out[0][i]
    boxes = out[1][i]
    scores = out[2][i]
    cls_id = out[3][i]
    image = origin_RGB[i]
    img_h, img_w = image.shape[:2]
    ratio, dwdh = resize_data[i][1:]
    for num in range(obj_num[0]):
        box = boxes[num]
        score = round(float(scores[num]), 3)
        obj_name = names[int(cls_id[num])]
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(img_w, box[2])
        y2 = min(img_h, box[3])
        color = colors[obj_name]
        obj_name += ' ' + str(score)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, obj_name, (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [225, 255, 255], thickness=2)
# %%
Image.fromarray(origin_RGB[0]).show()
# %%
Image.fromarray(origin_RGB[1]).show()
# %%
Image.fromarray(origin_RGB[2]).show()
