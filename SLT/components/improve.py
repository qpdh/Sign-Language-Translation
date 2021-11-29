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
        self.initUI()
        # self.capture = VideoCapture(self.cam_me)
        print('생성자 호출 완료')

    def initUI(self):
        # 700*800 크기 고정
        self.setFixedSize(900, 800)

        # 버튼에 기능을 연결하는 코드
        self.push_btn_number_setting.clicked.connect(self.buttonNumberSettingFunction)
        self.push_btn_english_setting.clicked.connect(self.buttonEnglishSettingFunction)
        self.push_btn_recognize.clicked.connect(self.buttonRecognizeFunction)
        self.push_btn_close.clicked.connect(self.buttonCloseFunction)

        # self.capture.start()

        # 숫자버튼 클릭
        self.pushButton_0.clicked.connect(self.ButtonListener_0)
        self.pushButton_1.clicked.connect(self.ButtonListener_1)
        self.pushButton_2.clicked.connect(self.ButtonListener_2)
        self.pushButton_3.clicked.connect(self.ButtonListener_3)
        self.pushButton_4.clicked.connect(self.ButtonListener_4)
        self.pushButton_5.clicked.connect(self.ButtonListener_5)
        self.pushButton_6.clicked.connect(self.ButtonListener_6)
        self.pushButton_7.clicked.connect(self.ButtonListener_7)
        self.pushButton_8.clicked.connect(self.ButtonListener_8)
        self.pushButton_9.clicked.connect(self.ButtonListener_9)

        # 영어 버튼 클릭
        self.pushButton_a.clicked.connect(self.ButtonListener_a)
        self.pushButton_b.clicked.connect(self.ButtonListener_b)
        self.pushButton_c.clicked.connect(self.ButtonListener_c)
        self.pushButton_d.clicked.connect(self.ButtonListener_d)
        self.pushButton_e.clicked.connect(self.ButtonListener_e)
        self.pushButton_f.clicked.connect(self.ButtonListener_f)
        self.pushButton_g.clicked.connect(self.ButtonListener_g)
        self.pushButton_h.clicked.connect(self.ButtonListener_h)
        self.pushButton_i.clicked.connect(self.ButtonListener_i)
        self.pushButton_j.clicked.connect(self.ButtonListener_j)
        self.pushButton_k.clicked.connect(self.ButtonListener_k)
        self.pushButton_l.clicked.connect(self.ButtonListener_l)
        self.pushButton_m.clicked.connect(self.ButtonListener_m)
        self.pushButton_n.clicked.connect(self.ButtonListener_n)
        self.pushButton_o.clicked.connect(self.ButtonListener_o)
        self.pushButton_p.clicked.connect(self.ButtonListener_p)
        self.pushButton_q.clicked.connect(self.ButtonListener_q)
        self.pushButton_r.clicked.connect(self.ButtonListener_r)
        self.pushButton_s.clicked.connect(self.ButtonListener_s)
        self.pushButton_t.clicked.connect(self.ButtonListener_t)
        self.pushButton_u.clicked.connect(self.ButtonListener_u)
        self.pushButton_v.clicked.connect(self.ButtonListener_v)
        self.pushButton_w.clicked.connect(self.ButtonListener_w)
        self.pushButton_x.clicked.connect(self.ButtonListener_x)
        self.pushButton_y.clicked.connect(self.ButtonListener_y)
        self.pushButton_z.clicked.connect(self.ButtonListener_z)

    # 0~9버튼 클릭
    def pushButton_0(self):
        print('0 button clicked')

    def buttonNumberSettingFunction(self):
        print("숫자 버튼이 눌렸습니다.")

    def buttonEnglishSettingFunction(self):
        print("영어 버튼이 눌렸습니다.")

    def buttonRecognizeFunction(self):
        print("재인식 버튼이 눌렸습니다.")

    def buttonCloseFunction(self):
        print("종료합니다.")
        self.deleteLater()

    # 숫자 버튼 리스너
    def ButtonListener_0(self):
        print("0 버튼이 눌렸습니다.")

    def ButtonListener_1(self):
        print("1 버튼이 눌렸습니다.")

    def ButtonListener_2(self):
        print("2 버튼이 눌렸습니다.")

    def ButtonListener_3(self):
        print("3 버튼이 눌렸습니다.")

    def ButtonListener_4(self):
        print("4 버튼이 눌렸습니다.")

    def ButtonListener_5(self):
        print("5 버튼이 눌렸습니다.")

    def ButtonListener_6(self):
        print("6 버튼이 눌렸습니다.")

    def ButtonListener_7(self):
        print("7 버튼이 눌렸습니다.")

    def ButtonListener_8(self):
        print("8 버튼이 눌렸습니다.")

    def ButtonListener_9(self):
        print("9 버튼이 눌렸습니다.")

    # 영어 버튼 리스너
    def ButtonListener_a(self):
        print("a 버튼이 눌렸습니다.")

    def ButtonListener_b(self):
        print("b 버튼이 눌렸습니다.")

    def ButtonListener_c(self):
        print("c 버튼이 눌렸습니다.")

    def ButtonListener_d(self):
        print("d 버튼이 눌렸습니다.")

    def ButtonListener_e(self):
        print("e 버튼이 눌렸습니다.")

    def ButtonListener_f(self):
        print("f 버튼이 눌렸습니다.")

    def ButtonListener_g(self):
        print("g 버튼이 눌렸습니다.")

    def ButtonListener_h(self):
        print("h 버튼이 눌렸습니다.")

    def ButtonListener_i(self):
        print("i 버튼이 눌렸습니다.")

    def ButtonListener_j(self):
        print("j 버튼이 눌렸습니다.")

    def ButtonListener_k(self):
        print("k 버튼이 눌렸습니다.")

    def ButtonListener_l(self):
        print("l 버튼이 눌렸습니다.")

    def ButtonListener_m(self):
        print("m 버튼이 눌렸습니다.")

    def ButtonListener_n(self):
        print("n 버튼이 눌렸습니다.")

    def ButtonListener_o(self):
        print("o 버튼이 눌렸습니다.")

    def ButtonListener_p(self):
        print("p 버튼이 눌렸습니다.")

    def ButtonListener_q(self):
        print("q 버튼이 눌렸습니다.")

    def ButtonListener_r(self):
        print("r 버튼이 눌렸습니다.")

    def ButtonListener_s(self):
        print("s 버튼이 눌렸습니다.")

    def ButtonListener_t(self):
        print("t 버튼이 눌렸습니다.")

    def ButtonListener_u(self):
        print("u 버튼이 눌렸습니다.")

    def ButtonListener_v(self):
        print("v 버튼이 눌렸습니다.")

    def ButtonListener_w(self):
        print("w 버튼이 눌렸습니다.")

    def ButtonListener_x(self):
        print("x 버튼이 눌렸습니다.")

    def ButtonListener_y(self):
        print("y 버튼이 눌렸습니다.")

    def ButtonListener_z(self):
        print("z 버튼이 눌렸습니다.")

    def showModal(self):
        return super().exec_()
