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

        # 언어변경
        self.pushButton_engNum.clicked.connect(self.changeLang)

        # 통화 연결
        self.pushButton_call.clicked.connect(self.call)

        # 설정
        self.pushButton_setting.clicked.connect(self.settingButtonListener)

    # 언어변경
    def changeLang(self):
        eng_num()
        print('changeLang clicked')

    # 통화 연결
    def call(self):
        global running
        if running:
            stop(self.textEdit_me)
        else:
            start()
        print('call clicked')

    # 설정
    def settingButtonListener(self):
        stop(self.textEdit_me)
        win = SettingDialog()
        win.showModal()

        print('setting button clicked')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    start(myWindow.cam_you, myWindow.textEdit_me)
    app.exec_()
