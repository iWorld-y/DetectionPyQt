import logging
import cv2
import numpy as np
import onnxruntime


class DetectV5:
    def __init__(self, onnx_session, classes):
        self.onnx_session: onnxruntime.InferenceSession = onnx_session
        self.classes = classes
        # 检出物数组
        self.result = []
        # 检出物数置信度
        self.score = []
        self.model_input_names = self.get_model_input_names()
        self.model_output_names = self.get_model_output_names()

    def inference(self, image: np.ndarray,
                  conf_thres: float = 0.4, iou_thres: float = 0.4):
        """
        对输入图像进行目标检测

        :param image: numpy.ndarray，输入的图像，形状为 (height, width, channels)，channels 为 3
        :param conf_thres: 置信度
        :param iou_thres: 交并比
        :return: numpy.ndarray，目标检测结果，形状为 (num_boxes, 6)，其中 num_boxes 表示检测出的目标数量，6 表示每个目标的参数数量（左上角坐标、右下角坐标、置信度和类别）
        """
        # 保存原图
        image_origin: np.ndarray = image.copy()
        # 预处理图像
        image = self.preprocess_image(image)
        # 构造输入数据
        input_feed = self.get_input_feed(image)
        # 执行推理, 得到边界框坐标、目标类别和置信度等信息
        prediction = self.onnx_session.run(None, input_feed)[0]
        # 使用非极大值抑制来过滤框
        prediction = self.filter_box(prediction, conf_thres, iou_thres)
        self.draw(image_origin, prediction)
        # image_origin = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        return image_origin

    def draw(self, image, box_data):
        #   取整，方便画框
        boxes = box_data[..., :4].astype(np.int32)
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)

        img_height_o = image.shape[0]
        img_width_o = image.shape[1]
        x_ratio = img_width_o / 640
        y_ratio = img_height_o / 640
        # 清空检出物数组与置信度数组
        self.result = []
        self.score = []
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box

            top = int(top * x_ratio)
            right = int(right * x_ratio)
            left = int(left * y_ratio)
            bottom = int(bottom * y_ratio)
            self.result.append(self.classes[cl])
            self.score.append(score)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.classes[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    def filter_box(self, origin_box, conf_thres: float, iou_thres: float):
        """
            过滤掉无用的框，删除置信度小于 conf_thres 的框，并将框的类别替换为置信度最大的类别，最后对每个类别进行非极大抑制。

            :param origin_box: numpy.ndarray，原始的检测框，形状为 (num_boxes, 6) 或 (1, num_boxes, 6)
            :param conf_thres: float，置信度阈值，小于此阈值的框将被过滤掉
            :param iou_thres: float，IoU 阈值，与其他框的 IoU 大于此阈值的框将被过滤掉
            :return: numpy.ndarray，经过过滤和非极大抑制后的检测框，形状为 (num_boxes, 6) 或 (1, num_boxes, 6)
        """
        origin_box = np.squeeze(origin_box)
        conf = origin_box[..., 4] > conf_thres
        box = origin_box[conf == True]
        #   通过argmax获取置信度最大的类别
        cls_cinf = box[..., 5:]
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))
        all_cls = list(set(cls))
        #   分别对每个类别进行过滤
        #   1.将第6列元素替换为类别下标
        #   2.xywh2xyxy 坐标转换
        #   3.经过非极大抑制后输出的BOX下标
        #   4.利用下标取出非极大抑制后的BOX
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])
            curr_cls_box = np.array(curr_cls_box)
            curr_cls_box = self.xywh2xyxy(curr_cls_box)
            curr_out_box = self.nms(curr_cls_box, iou_thres)
            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    @staticmethod
    def xywh2xyxy(x):
        """
        将坐标形式从 [x, y, w, h] 转换为 [x1, y1, x2, y2]

        :param x: numpy.ndarray，形状为 (num_boxes, 4) 或 (1, num_boxes, 4)
        :return: numpy.ndarray，形状为 (num_boxes, 4) 或 (1, num_boxes, 4)
        """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    @staticmethod
    def nms(dets, thresh):
        """
        非极大值抑制（Non-Maximum Suppression，NMS），用于去除重叠的检测框。

        :param dets: numpy.ndarray，形状为 (num_boxes, 6) 或 (1, num_boxes, 6)，每行包含 6 个元素，分别为 x1, y1, x2, y2, score, class
        :param thresh: float，IoU 阈值，与其他框的 IoU 大于此阈值的框将被过滤掉
        :return: list，经过非极大抑制后保留下来的框的下标
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        #   计算框的面积
        #   置信度从大到小排序
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]
            keep.append(i)
            #   计算相交面积
            #   1.相交
            #   2.不相交
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #   IOU小于thresh的框保留下来
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    @staticmethod
    def preprocess_image(image, target_size=(640, 640)):
        """
        对输入图像进行预处理

        :param image: numpy.ndarray，输入的图像，形状为 (height, width, channels)，channels 为 3
        :param target_size: tuple，目标尺寸，默认为 (640, 640)
        :return: numpy.ndarray，预处理后的图像，形状为 (1, channels, target_height, target_width)
        """
        # 将图像缩放到目标尺寸
        image = cv2.resize(image, target_size)
        # 将通道顺序反转并转置，以符合模型的输入格式
        image = image[:, :, ::-1].transpose(2, 0, 1)
        # 将图像类型转换为 float32，并归一化到 [0, 1]
        image = image.astype(dtype=np.float32) / 255.0
        # 将图像扩展为 batch 大小为 1 的数组
        image = np.expand_dims(image, axis=0)
        return image

    def get_input_feed(self, image_tensor):
        """

        :param image_tensor:
        :return:
        """
        input_feed = dict()
        for name in self.model_input_names:
            input_feed[name] = image_tensor
        return input_feed

    def get_model_input_names(self):
        """
        获取ONNX模型的输入节点名称列表。

        返回值：
            input_name: 包含所有输入节点名称的列表。
        """
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_model_output_names(self):
        """
        获取ONNX模型的输出节点名称列表。

        返回值：
            output_name: 包含所有输出节点名称的列表。
        """
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
