import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from setting import *
from video_cap import *

import threading
import mediapipe as mp
import cv2
import time
import numpy as np
from PyQt5 import QtGui

#import tensorflow

from_class = uic.loadUiType('main.ui')[0]

# 화면을 띄우는데 사용되는 class 선언
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.initUI()

    def initUI(self):
        self.setFixedSize(700, 800)

        start(self.cam_you)

        # 언어변경
        self.pushButton_engNum.clicked.connect(self.changeLang)

        # 통화 연결
        self.pushButton_call.clicked.connect(self.call)

        # 설정
        self.pushButton_setting.clicked.connect(self.settingButtonListener)

    # 언어변경
    def changeLang(self):
        i = 'eng'
        print(i, 'changeLang clicked')

    # 통화 연결
    def call(self):
        print('call clicked')

    # 설정
    def settingButtonListener(self):
        win = SettingDialog()
        win.showModal()
        print('insert button clicked')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
