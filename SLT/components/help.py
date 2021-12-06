from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore

from_class = uic.loadUiType('./ui/help.ui')[0]

# 화면을 띄우는데 사용되는 class 선언
class PicDialog(QDialog, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

        # 적용
        self.pushButton_ok.clicked.connect(self.okButtonListener)

    def initUI(self):
        self.setFixedSize(680, 700)

        # self.pushButton_cancel.clicked.connect(QtCore.QCoreApplication.instance().quit())

        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

    def okButtonListener(self):
        self.deleteLater()


    def showModal(self):
        return super().exec_()


