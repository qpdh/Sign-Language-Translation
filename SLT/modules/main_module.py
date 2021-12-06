from SLT.components.improve import *
from SLT.components.setting import *
from SLT.components.help import *

from_class = uic.loadUiType('./ui/main.ui')[0]


# 화면을 띄우는데 사용되는 class 선언
class MainModule(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.setFixedSize(700, 800)

        # 언어변경
        self.pushButton_engNum.clicked.connect(self.changeLang)

        # 통화 연결
        self.pushButton_call.clicked[bool].connect(self.call)

        # 설정
        self.pushButton_setting.clicked.connect(self.settingButtonListener)

        # 인식개선
        self.pushButton_improve.clicked.connect(self.improveButtonListener)

        # 지화보기
        self.pushButton_help.clicked.connect(self.pictureButtonListener)

    # 언어변경
    def changeLang(self):
        self.my_video_cap.eng_num()
        print('changeLang clicked')

    # 통화 연결 
    # main_server
    # main_client 
    # 에서 구현할 것
    def call(self):
        pass

    # 설정
    def settingButtonListener(self):
        win = SettingDialog()
        win.showModal()
        print('setting button clicked')

    # 인식개선
    def improveButtonListener(self):
        win = ImproveDialog()
        win.showModal()
        print('improve button clicked')

    # 지화보기
    def pictureButtonListener(self):
        win = PicDialog()
        win.showModal()
        print('pic button clicked')

    def show(self):
        super().show()
