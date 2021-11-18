import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

# UI 파일 연결
# 단, UI파일은 Python코드 파일과 같은 디렉토리에 위치해야 한다.
from_class = uic.loadUiType('인식 개선.ui')[0]


# 화면을 띄우는데 사용되는 class 선언
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 버튼에 기능을 연결하는 코드
        self.push_btn_number_setting.clicked.connect(self.buttonNumberSettingFunction)
        self.push_btn_english_setting.clicked.connect(self.buttonEnglishSettingFunction)
        self.push_btn_recognize.clicked.connect(self.buttonRecognizeFunction)
        self.push_btn_close.clicked.connect(self.buttonCloseFunction)

    def buttonNumberSettingFunction(self):
        print("숫자 버튼이 눌렸습니다.")

    def buttonEnglishSettingFunction(self):
        print("영어 버튼이 눌렸습니다.")

    def buttonRecognizeFunction(self):
        print("재인식 버튼이 눌렸습니다.")

    def buttonCloseFunction(self):
        print("종료합니다.")
        exit()


if __name__ == "__main__":
    # QApplication 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트 루프토 진입시키는(프로그램을 작동시키는)코드
    app.exec_()
