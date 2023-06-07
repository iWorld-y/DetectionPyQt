import time
import cv2
import numpy as np
import onnxruntime

from .utils import xywh2xyxy, nms, draw_detections


class DetectV8:
    # def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
    #     self.conf_threshold = conf_thres
    #     self.iou_threshold = iou_thres
    #
    #     # Initialize model
    #     self.initialize_model(path)
    def __init__(self, onnx_session, classes):
        self.onnx_session: onnxruntime.InferenceSession = onnx_session
        self.classes = classes
        # 检出物数组
        self.result = []
        # 检出物数置信度
        self.score = []
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.__inference__(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def __inference__(self, input_tensor):
        outputs = self.onnx_session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):
        # Rescale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    # def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
    #     return draw_detections(image, self.boxes, self.scores,
    #                            self.class_ids, mask_alpha)
    def inference(self, image, conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.detect_objects(image)
        # 记录检出物及其分数
        self.result = []
        self.score = []
        for ret, sco in zip(self.class_ids, self.scores):
            self.result.append(self.classes[ret])
            self.score.append(sco)
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids)

    def get_input_details(self):
        model_inputs = self.onnx_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.onnx_session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
