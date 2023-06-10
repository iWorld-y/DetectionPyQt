import logging
import os
import sys
import time
import cv2
import onnxruntime
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from src.ui.MainWindow import *
from src.YOLO_ONNX_Detection.DetectV5 import DetectV5
from src.YOLO_ONNX_Detection.DetectV6 import DetectV6
from src.YOLO_ONNX_Detection.DetectV8 import DetectV8
from src.YOLO_ONNX_Detection.GetDetector import GetDetector


class Gene_Window(QMainWindow, Ui_MainWindow):
    CLASSES = ["short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling",
               "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress",
               "sling dress"]  # DeepFashion2 13 类别

    def __init__(self, parent=None):
        super(Gene_Window, self).__init__(parent)

        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
        self.iou = 0.5
        self.conf = 0.5
        self.init_clicked()
        self.init_slider()

        self.init_onnx()
        self.choice_onnx()

    def choice_onnx(self):
        self.main_ui.YOLOv5.toggled.connect(self.get_detect_v5)
        self.main_ui.YOLOv6.toggled.connect(self.get_detect_v6)
        self.main_ui.YOLOv8.toggled.connect(self.get_detect_v8)

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
        # 检测图片
        self.main_ui.detect_image.clicked.connect(self.detect_image)
        # 检测视频
        self.main_ui.detect_video.clicked.connect(self.load_video)
        # 暂停摄像头画面
        self.main_ui.pause_video.clicked.connect(self.toggle_pause)
        # 检测摄像头
        self.main_ui.detect_camer.clicked.connect(self.open_camer)

        # 标识当前是否处于暂停状态
        self.paused_camer = False
        # 退出
        self.main_ui.quit_button.clicked.connect(QApplication.quit)

    def init_onnx(self):
        # 初始化模型
        ONNX_path = "../models/"
        for version in [5, 6, 8]:
            if (not os.path.isfile(os.path.join(ONNX_path, f"v{version}.onnx"))):
                QtWidgets.QMessageBox.warning(self, "错误", "权重加载失败\n请检查 models 目录",
                                              buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
                raise ValueError(f"ONNX model file not found at {ONNX_path}")
        # 将所有权重加载到内存中
        self.detectors = GetDetector(self.CLASSES, "../models").get_detectors()
        # 默认权重为 yolov5
        self.main_ui.YOLOv5.setChecked(True)
        self.get_detect_v5()

    def get_detect_v5(self):
        if (not self.main_ui.YOLOv5.isChecked()):
            # 避免取消选中时也被调用
            return
        logging.info("当前权重：v5")
        self.detector = self.detectors[0]

    def get_detect_v6(self):
        if (not self.main_ui.YOLOv6.isChecked()):
            return
        logging.info("当前权重：v6")
        self.detector = self.detectors[1]

    def get_detect_v8(self):
        if (not self.main_ui.YOLOv8.isChecked()):
            return
        logging.info("当前权重：v8")
        self.detector = self.detectors[2]

    def toggle_pause(self):
        # 处理暂停信号
        self.paused_camer = not self.paused_camer
        self.main_ui.pause_video.setText(
            QtCore.QCoreApplication.translate("MainWindow", "暂停" if self.paused_camer else "继续"))

    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "打开视频",
                                                         "../assert",
                                                         "All Files(*)")
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, "错误", "未选择视频", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            self.main_ui.detect_video.setText(
                QtCore.QCoreApplication.translate("MainWindow", "选择权重"))
            return
        self.main_ui.detect_video.setText(
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
            start_time = time.time()
            origin_stream = self.detector.inference(video_stream, self.conf, self.iou)
            end_time = time.time()
            self.main_ui.FPS.setText(f"FPS: {1.0 / (end_time - start_time):.2f}")
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
        # 显示检出物及其分数
        for ret, sco in zip(self.detector.result, self.detector.score):
            self.main_ui.detect_result_text.append(f"{ret}: {sco:.2f}")

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
            start_time = time.time()
            origin_stream = self.detector.inference(video_stream, self.conf, self.iou)
            end_time = time.time()
            self.main_ui.FPS.setText(f"FPS: {1.0 / (end_time - start_time):.2f}")
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
        # 显示检出物及其分数
        for ret, sco in zip(self.detector.result, self.detector.score):
            self.main_ui.detect_result_text.append(f"{ret}: {sco:.2f}")

    def detect_image(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片",
                                                            "../assert",
                                                            "*.jpg;;*.png;;All Files(*)")
        if (not self.imgName):
            QtWidgets.QMessageBox.warning(self, "错误", "未选择图片", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
            return
        # 清空 FPS 显示
        self.main_ui.FPS.setText('')

        jpg = QtGui.QPixmap(self.imgName).scaled(self.main_ui.origin_image.width(), self.main_ui.origin_image.height())
        self.main_ui.origin_image.setPixmap(jpg)
        try:
            image = self.detector.inference(cv2.imread(self.imgName), self.conf, self.iou)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            image = cv2.cvtColor(cv2.imread(self.imgName), cv2.COLOR_BGR2RGB)
            logging.error(e)
        image = QImage(image[:], image.shape[1], image.shape[0], image.shape[1] * 3,
                       QImage.Format_RGB888)
        pixmap_imgSrc = QtGui.QPixmap.fromImage(image).scaled(self.main_ui.show_label.width(),
                                                              self.main_ui.show_label.height())
        self.main_ui.show_label.setPixmap(pixmap_imgSrc)
        # 显示检出物及其分数
        for ret, sco in zip(self.detector.result, self.detector.score):
            self.main_ui.detect_result_text.append(f"{ret}: {sco:.2f}")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)
    app = QApplication(sys.argv)
    myWin = Gene_Window()
    myWin.show()
    sys.exit(app.exec_())
