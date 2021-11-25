import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

from_class = uic.loadUiType('setting.ui')[0]

# 화면을 띄우는데 사용되는 class 선언
class DepositDialog(QDialog, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

        # DB 연결 클래스 호출
        # self.drinkManagement = DrinkDBManagement()
        #
        # # pushButton 리스트
        # self.pushButtonList = [self.pushButton_drink0, self.pushButton_drink1, self.pushButton_drink2,
        #                        self.pushButton_drink3,
        #                        self.pushButton_drink4, self.pushButton_drink5, self.pushButton_drink6,
        #                        self.pushButton_drink7]
        # # 음료수 명 리스트
        # self.drinkNameList = [self.label_name0, self.label_name1, self.label_name2, self.label_name3,
        #                       self.label_name4, self.label_name5, self.label_name6, self.label_name7]
        #
        # # 음료수 가격 리스트
        # self.drinkCostList = [self.label_cost0, self.label_cost1, self.label_cost2, self.label_cost3,
        #                       self.label_cost4, self.label_cost5, self.label_cost6, self.label_cost7]

    def initUI(self):
        # 140*240 크기 고정
        self.setFixedSize(140, 240)
    #     self.bringName_Cost()
    #
    #
    # def bringName_Cost(self, drinkNameList, drinkCostList):
    #     drinksInfo = self.drinkManagement.print_all_drink()
    #
    #     # todo drinkInfo & drinkNameList & drinkCostList 길이가 같아야 함
    #     for i in range(len(drinksInfo)):
    #         name = drinksInfo[i]["name"]
    #         cost = drinksInfo[i]["cost"]
    #         drinkNameList[i].setText(name)
    #         drinkCostList[i].setText(str(cost))

    def showModal(self):
        return super().exec_()