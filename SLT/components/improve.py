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
        #self.capture = VideoCapture(self.cam_me)
        print('생성자 호출 완료')

    def initUI(self):
        # 700*800 크기 고정
        self.setFixedSize(900, 800)

        # 버튼에 기능을 연결하는 코드
        self.push_btn_number_setting.clicked.connect(self.buttonNumberSettingFunction)
        self.push_btn_english_setting.clicked.connect(self.buttonEnglishSettingFunction)
        self.push_btn_recognize.clicked.connect(self.buttonRecognizeFunction)
        self.push_btn_close.clicked.connect(self.buttonCloseFunction)


        # 숫자버튼 클릭
        self.pushButton_0.clicked.connect(self.pushButton_0)
        self.pushButton_1.clicked.connect(self.pushButton_1)
        self.pushButton_2.clicked.connect(self.pushButton_2)
        self.pushButton_3.clicked.connect(self.pushButton_3)
        self.pushButton_4.clicked.connect(self.pushButton_4)
        self.pushButton_5.clicked.connect(self.pushButton_5)
        self.pushButton_6.clicked.connect(self.pushButton_6)
        self.pushButton_7.clicked.connect(self.pushButton_7)
        self.pushButton_8.clicked.connect(self.pushButton_8)
        self.pushButton_9.clicked.connect(self.pushButton_9)

        # 영어버튼 클릭
        self.pushButton_a.clicked.connect(self.pushButton_a)
        self.pushButton_b.clicked.connect(self.pushButton_b)
        self.pushButton_c.clicked.connect(self.pushButton_c)
        self.pushButton_d.clicked.connect(self.pushButton_d)
        self.pushButton_e.clicked.connect(self.pushButton_e)
        self.pushButton_f.clicked.connect(self.pushButton_f)
        self.pushButton_g.clicked.connect(self.pushButton_g)
        self.pushButton_h.clicked.connect(self.pushButton_h)
        self.pushButton_i.clicked.connect(self.pushButton_i)
        self.pushButton_j.clicked.connect(self.pushButton_j)
        self.pushButton_k.clicked.connect(self.pushButton_k)
        self.pushButton_l.clicked.connect(self.pushButton_l)
        self.pushButton_m.clicked.connect(self.pushButton_m)
        self.pushButton_n.clicked.connect(self.pushButton_n)
        self.pushButton_o.clicked.connect(self.pushButton_o)
        self.pushButton_p.clicked.connect(self.pushButton_p)
        self.pushButton_q.clicked.connect(self.pushButton_q)
        self.pushButton_r.clicked.connect(self.pushButton_r)
        self.pushButton_s.clicked.connect(self.pushButton_s)
        self.pushButton_t.clicked.connect(self.pushButton_t)
        self.pushButton_u.clicked.connect(self.pushButton_u)
        self.pushButton_v.clicked.connect(self.pushButton_v)
        self.pushButton_w.clicked.connect(self.pushButton_w)
        self.pushButton_x.clicked.connect(self.pushButton_x)
        self.pushButton_y.clicked.connect(self.pushButton_y)
        self.pushButton_z.clicked.connect(self.pushButton_z)

        #self.capture.start()

    def buttonNumberSettingFunction(self):
        print("숫자 버튼이 눌렸습니다.")

    def buttonEnglishSettingFunction(self):
        print("영어 버튼이 눌렸습니다.")

    def buttonRecognizeFunction(self):
        print("재인식 버튼이 눌렸습니다.")

    def buttonCloseFunction(self):
        print("종료합니다.")
        self.deleteLater()

    #0~9버튼 클릭
    def pushButton_0(self):
        print('0 button clicked')
    def pushButton_1(self):
        print('1 button clicked')
    def pushButton_2(self):
        print('2 button clicked')
    def pushButton_3(self):
        print('3 button clicked')
    def pushButton_4(self):
        print('4 button clicked')
    def pushButton_5(self):
        print('5 button clicked')
    def pushButton_6(self):
        print('6 button clicked')
    def pushButton_7(self):
        print('7 button clicked')
    def pushButton_8(self):
        print('8 button clicked')
    def pushButton_9(self):
        print('9 button clicked')


    #a~z버튼 클릭
    def pushButton_a(self):
        print('a button clicked')
    def pushButton_b(self):
        print('b button clicked')
    def pushButton_c(self):
        print('c button clicked')
    def pushButton_d(self):
        print('d button clicked')
    def pushButton_e(self):
        print('e button clicked')
    def pushButton_f(self):
        print('f button clicked')
    def pushButton_g(self):
        print('g button clicked')
    def pushButton_h(self):
        print('h button clicked')
    def pushButton_i(self):
        print('i button clicked')
    def pushButton_j(self):
        print('j button clicked')
    def pushButton_k(self):
        print('k button clicked')
    def pushButton_l(self):
        print('l button clicked')
    def pushButton_m(self):
        print('m button clicked')
    def pushButton_n(self):
        print('n button clicked')
    def pushButton_o(self):
        print('o button clicked')
    def pushButton_p(self):
        print('p button clicked')
    def pushButton_q(self):
        print('q button clicked')
    def pushButton_r(self):
        print('r button clicked')
    def pushButton_s(self):
        print('s button clicked')
    def pushButton_t(self):
        print('t button clicked')
    def pushButton_u(self):
        print('u button clicked')
    def pushButton_v(self):
        print('v button clicked')
    def pushButton_w(self):
        print('w button clicked')
    def pushButton_x(self):
        print('x button clicked')
    def pushButton_y(self):
        print('y button clicked')
    def pushButton_z(self):
        print('z button clicked')



    def showModal(self):
        return super().exec_()
