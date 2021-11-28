from setting import *
from video_cap import *
from PyQt5 import QtGui
import sys
import socket_module

from_class = uic.loadUiType('main.ui')[0]


# 화면을 띄우는데 사용되는 class 선언
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

        # 서버 소켓 생성
        self.my_socket = socket_module.server_socket()

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
        self.my_video_cap.eng_num()
        print('changeLang clicked')

    # 통화 연결
    def call(self):
        print(self.my_video_cap.running)

        if self.my_video_cap.running:
            self.textEdit_me.clear()
            self.cam_you.setPixmap(QtGui.QPixmap('stop.png'))
            self.my_video_cap.stop()
        else:
            print('main.call (running state : false)')
            self.my_video_cap.start()

    # 설정
    def settingButtonListener(self):
        self.my_video_cap.stop()
        win = SettingDialog()
        win.showModal()

        print('setting button clicked')

    def show(self):
        super().show()
        self.my_video_cap = video_cap(self.my_socket, self.cam_me, self.cam_you, self.textEdit_me, self.textEdit_you )
        self.my_video_cap.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
