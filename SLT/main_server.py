from PyQt5 import QtGui
import sys
from SLT.modules import socket_module
from SLT.components.help import *
from SLT.modules.main_module import MainModule
from SLT.modules.video_cap_server import VideoCaptureServer


# 화면을 띄우는데 사용되는 class 선언
class WindowClass(MainModule):
    def __init__(self):
        super().__init__()

    # 통화 연결
    def call(self):
        # 서버 소켓 생성
        # print(self.my_video_cap.running)

        print('test')
        my_socket = socket_module.server_socket()

        # if self.my_video_cap.running:
        #     self.textEdit_me.clear()
        #     self.cam_you.setPixmap(QtGui.QPixmap('images/stop.png'))
        #     self.my_video_cap.stop()
        # else:
        #     print('main.call (running state : false)')
        #     self.my_video_cap.start()

        my_video_cap = VideoCaptureServer(self.cam_me, self.cam_you, self.textEdit_me, self.textEdit_you, my_socket)
        my_video_cap.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
