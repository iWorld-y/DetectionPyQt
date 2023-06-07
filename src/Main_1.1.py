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
from DetectV5 import DetectV5


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

    def set_conf(self, value):
        self.main_ui.Conf_label.setText(f"Conf:\t{self.conf:.2f}")
        self.conf = value / 100

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
        # self.video_capture = cv2.VideoCapture(0)
        # 检测摄像头
        self.main_ui.detect_camer.clicked.connect(self.open_camer)

        # 标识当前是否处于暂停状态
        self.paused_camer = False
        # 退出
        self.main_ui.quit_button.clicked.connect(QApplication.quit)

    def __init_detect_v5__(self, ONNX_path):
        if (os.path.isfile(ONNX_path)):
            self.detect_v5 = DetectV5(onnxruntime.InferenceSession(ONNX_path, providers=['CPUExecutionProvider']),
                                      classes=self.CLASSES)
        else:
            raise ValueError(f"ONNX model file not found at {ONNX_path}")

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
        # 读取视频
        self.video_capture = cv2.VideoCapture(self.video_path)
        _, image = self.video_capture.read()
        cv2.imwrite("test.jpg", image)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_video)
        self.timer.start(50)

    def detect_video(self):
        # 如果处于暂停状态，直接返回
        if self.paused_camer:
            return

        # 获取一帧画面
        # read()：读取视频流中的一帧，返回两个值，第一个值是一个布尔值，表示是否成功读取了一帧；第二个值是一个 NumPy 数组，表示读取的图像数据。
        ret, video_stream = self.video_capture.read()
        height, width, channel = video_stream.shape

        # 预测画面
        try:
            origin_stream = self.detect_v5.inference(video_stream, self.conf, self.iou)
            # 矫正颜色
            origin_stream = cv2.cvtColor(origin_stream, cv2.COLOR_BGR2RGB)
            video_stream = cv2.cvtColor(video_stream, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.warning("未发现衣物")
            video_stream = cv2.cvtColor(video_stream, cv2.COLOR_BGR2RGB)
            origin_stream = video_stream

        # 创建 QImage 对象，将原画面显示出来
        qimage = QImage(video_stream, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimage).scaled(self.main_ui.origin_image.width(),
                                              self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(pixmap)
        # 将检查结果显示出来
        origin_stream = QImage(origin_stream[:], origin_stream.shape[1], origin_stream.shape[0],
                               origin_stream.shape[1] * 3, QImage.Format_RGB888)
        pixmap_stream = QtGui.QPixmap.fromImage(origin_stream).scaled(self.main_ui.show_label.width(),
                                                                      self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_stream)

    def open_camer(self):
        # 打开摄像头
        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.detect_camer)
        self.timer.start(50)

    def detect_camer(self):
        # 如果处于暂停状态，直接返回
        if self.paused_camer:
            return
        if (not self.video_capture.isOpened()):
            QtWidgets.QMessageBox.warning(self, "错误", "摄像头无法打开", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            return
        # 读取摄像头画面并将其翻转
        _, video_stream = self.video_capture.read()
        video_stream = cv2.flip(video_stream, 1)

        height, width, channel = video_stream.shape

        # 预测画面
        try:
            origin_stream = self.detect_v5.inference(video_stream, self.conf, self.iou)
            # 矫正颜色
            origin_stream = cv2.cvtColor(origin_stream, cv2.COLOR_BGR2RGB)
            video_stream = cv2.cvtColor(video_stream, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.warning("未发现衣物")
            video_stream = cv2.cvtColor(video_stream, cv2.COLOR_BGR2RGB)
            origin_stream = video_stream

        # 创建 QImage 对象，并从摄像头画面中获取像素数据
        qimage = QImage(video_stream, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(qimage).scaled(self.main_ui.origin_image.width(),
                                              self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(pixmap)
        # 显示检测结果
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
            self.__init_detect_v5__(self.weight_path)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "错误", "权重不存在",
                                          buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            logging.error(e)
            return
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "错误", "权重解析失败\n请检查权重是否正确",
                                          buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            logging.error(e)
            return
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
        try:
            image = self.detect_v5.inference(cv2.imread(self.imgName), self.conf, self.iou)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(e)
        image = QImage(image[:], image.shape[1], image.shape[0], image.shape[1] * 3,
                       QImage.Format_RGB888)
        pixmap_imgSrc = QtGui.QPixmap.fromImage(image).scaled(self.main_ui.show_label.width(),
                                                              self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_imgSrc)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    app = QApplication(sys.argv)
    myWin = Gene_Window()
    myWin.show()
    sys.exit(app.exec_())
