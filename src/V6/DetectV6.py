import time
import cv2
import numpy as np
import onnxruntime

from .utils import xywh2xyxy, nms, draw_detections


class DetectV6:

    # def __init__(self, path):
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

    def initialize_model(self, path):
        self.onnx_session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform __inference__ on the image
        output = self.__inference__(input_tensor)

        # Process output data
        self.boxes, self.scores, self.classes = self.process_output(output)

        return self.boxes, self.scores, self.classes

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
        start = time.perf_counter()
        outputs = self.onnx_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"__inference__ time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output)

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes /= np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def inference(self, image, conf_thres=0.5, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.detect_objects(image)
        return draw_detections(image, self.boxes, self.scores,
                               self.classes, mask_alpha=0)

    def get_input_details(self):
        model_inputs = self.onnx_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.onnx_session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':
    from PIL import Image

    ONNX_path = "../../models/v6.onnx"
    classes = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
               "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
               "sling dress"]
    DetectV6_detector = DetectV6(onnxruntime.InferenceSession(ONNX_path, providers=['CPUExecutionProvider']), classes)

    img = cv2.imread(r"/home/eugene/autodl-tmp/test/1.jpg")
    # Draw detections
    combined_img = DetectV6_detector.inference(img, conf_thres=0.3, iou_thres=0.5)
    Image.fromarray(combined_img).show()
