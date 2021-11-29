from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore
from SLT.modules.video_cap import VideoCapture

# UI 파일 연결
# 단, UI파일은 Python코드 파일과 같은 디렉토리에 위치해야 한다.
from_class = uic.loadUiType('./ui/improve.ui')[0]


# 화면을 띄우는데 사용되는 class 선언
class ImproveDialog(QDialog, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.capture = VideoCapture(self.cam_me)

        self.initUI()



    def initUI(self):
        # 700*800 크기 고정
        self.setFixedSize(900, 800)

        # 버튼에 기능을 연결하는 코드
        self.push_btn_number_setting.clicked.connect(self.buttonNumberSettingFunction)
        self.push_btn_english_setting.clicked.connect(self.buttonEnglishSettingFunction)
        self.push_btn_recognize.clicked.connect(self.buttonRecognizeFunction)
        self.push_btn_close.clicked.connect(self.buttonCloseFunction)

        print('캡쳐 시작')
        self.capture.start()
        print('캡쳐 끝')

    def buttonNumberSettingFunction(self):
        print("숫자 버튼이 눌렸습니다.")

    def buttonEnglishSettingFunction(self):
        print("영어 버튼이 눌렸습니다.")

    def buttonRecognizeFunction(self):
        print("재인식 버튼이 눌렸습니다.")

    def buttonCloseFunction(self):
        print("종료합니다.")
        self.deleteLater()

    def showModal(self):
        return super().exec_()
