import logging
import os
import sys

import cv2
import numpy as np
import onnxruntime
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from MainWindow import *


class Gene_Window(QMainWindow, Ui_MainWindow):
    CLASSES = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
               "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
               "sling dress"]  # coco80类别

    def __init__(self, parent=None):
        super(Gene_Window, self).__init__(parent)

        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.iou = 0.5
        self.conf = 0.5
        self.init_clicked()
        self.init_slider()

    def init_slider(self):
        # IoU
        self.main_ui.IoU_label.setText(f"IoU:\t{self.iou:.2f}")
        self.main_ui.IoU_Slider.setMinimum(1)
        self.main_ui.IoU_Slider.setMaximum(100)
        self.main_ui.IoU_Slider.setValue(int(self.iou * 100))
        self.main_ui.IoU_Slider.valueChanged[int].connect(self.set_iou)

        # Conf
        self.main_ui.Conf_label.setText(f"Conf:\t{self.conf:.2f}")
        self.main_ui.Conf_Slider.setMinimum(1)
        self.main_ui.Conf_Slider.setMaximum(100)
        self.main_ui.Conf_Slider.setValue(int(self.conf * 100))
        self.main_ui.Conf_Slider.valueChanged[int].connect(self.set_conf)

    def set_iou(self, value):
        self.main_ui.IoU_label.setText(f"IoU:\t{self.iou:.2f}")
        self.iou = value / 100
        logging.info(f"iou:{self.iou}")

    def set_conf(self, value):
        self.main_ui.Conf_label.setText(f"Conf:\t{self.conf:.2f}")
        self.conf = value / 100
        logging.info(f"Conf:{self.conf}")

    def init_clicked(self):
        # 打开权重
        self.main_ui.open_weight.clicked.connect(self.open_weight)
        # 检测图片
        self.main_ui.detect_image.clicked.connect(self.detect_image)
        # 检测视频
        self.main_ui.detect_video.clicked.connect(self.load_video)

        # 暂停摄像头画面
        self.main_ui.pause_video.clicked.connect(self.toggle_pause)

        # 初始化摄像头
        self.video_capture = cv2.VideoCapture(0)
        self.main_ui.detect_camer.clicked.connect(self.open_camer)

        # 标识当前是否处于暂停状态
        self.paused_camer = False
        # 退出
        self.main_ui.quit_button.clicked.connect(QApplication.quit)

    def toggle_pause(self):
        # 处理暂停信号
        self.paused_camer = not self.paused_camer
        self.main_ui.pause_video.setText(
            QtCore.QCoreApplication.translate("MainWindow", "暂停" if self.paused_camer else "继续"))

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "打开视频",
                                                         "/home/eugene/autodl-tmp/test",
                                                         "All Files(*)")
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "错误", "未选择视频", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.main_ui.open_weight.setText(
                QtCore.QCoreApplication.translate("MainWindow", "选择权重"))
            return
        self.main_ui.open_weight.setText(
            QtCore.QCoreApplication.translate("MainWindow", f"正在检测:\n{os.path.basename(self.video_path)}"))

        self.video_capture = cv2.VideoCapture(self.video_path)
        # 创建定时器，用于更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_video)
        self.timer.start(50)
        self.detect_video()

    def detect_video(self):
        # 如果处于暂停状态，直接返回
        if self.paused_camer:
            return

        _, self.video_stream = self.video_capture.read()
        # 矫正颜色
        self.video_stream = cv2.cvtColor(self.video_stream, cv2.COLOR_BGR2RGB)
        height, width, channel = self.video_stream.shape
        # 创建 QImage 对象，将原画面显示出来
        qimage = QImage(self.video_stream, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimage).scaled(self.main_ui.origin_image.width(),
                                              self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(pixmap)
        # 预测画面
        try:
            output_stream, origin_stream = self.inference('', self.video_stream)
            outbox_stream = self.filter_box(output_stream, self.conf, self.iou)
            self.draw(origin_stream, outbox_stream)
        except Exception as e:
            logging.warning("未发现衣物")
            origin_stream = self.video_stream
        origin_stream = QImage(origin_stream[:], origin_stream.shape[1], origin_stream.shape[0],
                               origin_stream.shape[1] * 3, QImage.Format_RGB888)
        pixmap_stream = QtGui.QPixmap.fromImage(origin_stream).scaled(self.main_ui.show_label.width(),
                                                                      self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_stream)

    def open_camer(self):
        # 创建定时器，用于更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.open_camer)
        self.timer.start(50)
        # 如果处于暂停状态，直接返回
        if self.paused_camer:
            return
        try:
            assert 1==0
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "错误", "摄像头无法打开", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        if (not self.video_capture.isOpened()):
            QtWidgets.QMessageBox.warning(self, "错误", "摄像头无法打开", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            return
        # 读取摄像头画面并将其翻转
        _, video_stream = self.video_capture.read()
        video_stream = cv2.flip(video_stream, 1)
        # 矫正颜色
        video_stream = cv2.cvtColor(video_stream, cv2.COLOR_BGR2RGB)
        height, width, channel = video_stream.shape
        # video_stream = self.inference('', video_stream)
        # 创建 QImage 对象，并从摄像头画面中获取像素数据
        qimage = QImage(video_stream, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimage).scaled(self.main_ui.origin_image.width(),
                                              self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(pixmap)

        # 预测画面
        try:
            output_stream, origin_stream = self.inference('', video_stream)
            outbox_stream = self.filter_box(output_stream, self.conf, self.iou)
            self.draw(origin_stream, outbox_stream)
        except Exception as e:
            logging.warning("未发现衣物")
            origin_stream = video_stream
        origin_stream = QImage(origin_stream[:], origin_stream.shape[1], origin_stream.shape[0],
                               origin_stream.shape[1] * 3, QImage.Format_RGB888)
        pixmap_stream = QtGui.QPixmap.fromImage(origin_stream).scaled(self.main_ui.show_label.width(),
                                                                      self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_stream)

    def open_weight(self):
        self.weight_path, _ = QFileDialog.getOpenFileName(self.main_ui.open_weight, "选择权重",
                                                          '/home/eugene/autodl-tmp/weights',
                                                          "*.onnx")
        if not self.weight_path:
            QtWidgets.QMessageBox.warning(self, "错误", "未选择权重", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.main_ui.open_weight.setText(
                QtCore.QCoreApplication.translate("MainWindow", "选择权重"))
            return

        weight_name = os.path.basename(self.weight_path)
        try:  # 尝试加载 ONNX 权重，若报错即权重不可用
            self.onnx_session = onnxruntime.InferenceSession(self.weight_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "错误", "权重解析失败\n请检查权重是否正确",
                                          buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            logging.error("权重解析失败，请检查权重是否正确")
            return
        logging.info(f"已获取 ONNX:\t{weight_name}")

        self.model_input_names = self.get_model_input_names()
        self.model_output_names = self.get_model_output_names()
        logging.info("ONNX 初始化完成")
        self.main_ui.open_weight.setText(
            QtCore.QCoreApplication.translate("MainWindow", f"当前权重：\n{weight_name}"))

    def detect_image(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片",
                                                            "/home/eugene/autodl-tmp/test",
                                                            "*.jpg;;*.png;;All Files(*)")
        if (not self.imgName):
            QtWidgets.QMessageBox.warning(self, "错误", "未选择图片", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            return
        jpg = QtGui.QPixmap(self.imgName).scaled(self.main_ui.origin_image.width(), self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(jpg)

        logging.info(f"已打开图片：{self.imgName}")
        logging.info(f"开始推理")
        output_img, origin_img = self.inference(self.imgName, np.ndarray([0]))
        outbox_img = self.filter_box(output_img, self.conf, self.iou)
        self.draw(origin_img, outbox_img)
        logging.info(f"推理完成")
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        origin_img = QImage(origin_img[:], origin_img.shape[1], origin_img.shape[0], origin_img.shape[1] * 3,
                            QImage.Format_RGB888)
        pixmap_imgSrc = QtGui.QPixmap.fromImage(origin_img).scaled(self.main_ui.show_label.width(),
                                                                   self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_imgSrc)

    def inference(self, img_path: str, img: np.ndarray):
        if (img_path):
            img = cv2.imread(img_path)
        img_o = img.copy()
        or_img = cv2.resize(img, (640, 640))
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        input_feed = self.get_input_feed(img)
        pred = self.onnx_session.run(None, input_feed)[0]
        return pred, img_o

    # dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
    # thresh: 阈值
    def nms(self, dets, thresh):
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

    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def filter_box(self, org_box, conf_thres, iou_thres):
        # 过滤掉无用的框
        #   删除为1的维度
        #   删除置信度小于conf_thres的BOX
        org_box = np.squeeze(org_box)
        conf = org_box[..., 4] > conf_thres
        box = org_box[conf == True]
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

    def draw(self, image, box_data):
        #   取整，方便画框
        boxes = box_data[..., :4].astype(np.int32)
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)

        img_height_o = image.shape[0]
        img_width_o = image.shape[1]
        x_ratio = img_width_o / 640
        y_ratio = img_height_o / 640

        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            self.main_ui.detect_result_text.append(f"检测到：[{self.CLASSES[cl]}]，分数为：{score:.2f}")
            self.main_ui.detect_result_text.append(f"检测框坐标：{top}, {left}, {right}, {bottom}\n")

            top = int(top * x_ratio)
            right = int(right * x_ratio)
            left = int(left * y_ratio)
            bottom = int(bottom * y_ratio)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    def get_input_feed(self, img_tensor):
        """
        输入图像
        """
        input_feed = {}
        for name in self.model_input_names:
            input_feed[name] = img_tensor
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


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    app = QApplication(sys.argv)
    myWin = Gene_Window()
    myWin.show()
    sys.exit(app.exec_())
