import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

from_class = uic.loadUiType('main.ui')[0]

# 화면을 띄우는데 사용되는 class 선언
class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.initUI()

    def initUI(self):
        # 900*500 크기 고정
        self.setFixedSize(700, 800)

        # 버튼에 기능을 연결하는 코드
        self.pushButton_engNum.clicked.connect(self.changeLang)

        # 구입
        self.pushButton_call.clicked.connect(self.call)

    # 제품 버튼 리스너
    def changeLang(self):
        i = 'eng'
        print(i, 'changeLang clicked')

    # 구매 버튼 리스너
    def call(self):
        print('call clicked')

    # 입금 버튼 리스너
    def settingButtonListener(self):
        win = DepositDialog()
        win.showModal()
        print('insert button clicked')

    # 잔돈 반환 버튼 리스너
    def returnButtonListener(self):
        print('return button clicked')

    # 관리자 모드 버튼 리스너
    def directorButtonListener(self):
        win = ManagerDialog()
        win.showModal()
        print('director button clicked')

if __name__ == "__main__":
    # QApplication 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스생성
    myWindow = WindowClass()

    # 프로그램 화면을 보여주는 코드
    myWindow.show()

    # 프로그램을 이벤트 루프토 진입시키는(프로그램을 작동시키는)코드
    app.exec_()
