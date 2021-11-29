from PyQt5 import QtGui
import sys
from SLT.modules import socket_module

from SLT.components.help import *
from modules.main_module import MainModule

from_class = uic.loadUiType('./ui/main.ui')[0]


# 화면을 띄우는데 사용되는 class 선언
class WindowClass(MainModule):
    def __init__(self):
        super().__init__()

    # 통화 연결
    def call(self):
        print(self.my_video_cap.running)

        # 클라이언트 소켓 생성
        self.my_socket = socket_module.client_socket()

        if self.my_video_cap.running:
            self.textEdit_me.clear()
            self.cam_you.setPixmap(QtGui.QPixmap('images/stop.png'))
            self.my_video_cap.stop()
        else:
            print('main.call (running state : false)')
            self.my_video_cap.start()

        # self.my_video_cap = video_cap(self.my_socket, self.cam_me, self.cam_you, self.textEdit_me, self.textEdit_you)
        # self.my_video_cap.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
