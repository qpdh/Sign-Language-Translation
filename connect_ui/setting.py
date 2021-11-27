import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore

from_class = uic.loadUiType('setting.ui')[0]

# 화면을 띄우는데 사용되는 class 선언
class SettingDialog(QDialog, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.setFixedSize(440, 220)

        # self.pushButton_cancel.clicked.connect(QtCore.QCoreApplication.instance().quit())

        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

    def showModal(self):
        return super().exec_()


