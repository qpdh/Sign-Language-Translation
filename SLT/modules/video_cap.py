import threading
import mediapipe as mp
import cv2
import time
import numpy as np
from PyQt5 import QtGui


class video_cap:
    SERVER = 0
    CLIENT = 1

    # self.my_socket, self.cam_me, self.cam_you, self.textEdit_me, self.textEdit_you
    def __init__(self, my_socket, cam_me, cam_you, editText_me, editText_you):
        self.th = None
        self.running = True
        self.isEng = True
        self.cam_me = cam_me
        self.cam_you = cam_you
        self.editText_me = editText_me
        self.editText_you = editText_you
        self.my_socket = my_socket
        self.my_image = None
        self.addr = None

    def capStart(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        recognizeDelay = 0.5
        start_time = time.time()
        premotion = 0

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        sentence = []
        actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        actions_alpha = [chr(i) for i in range(65, 91)]

        recvThread = threading.Thread(target=self.recv_thread)
        recvThread.start()

        # sendThread = threading.Thread(target=self.send_thread)
        # sendThread.start()

        # model = tensorflow.keras.models.load_model('./OutputModel_Alpha')

        def joint_to_angle(landmark):

            # 21 x,y,z np array 생성
            joint = np.zeros((21, 3))

            for j, lm in enumerate(landmark.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            joint1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            joint2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

            # 각 좌표의 벡터를 구함
            vec = joint1 - joint2
            vec = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]

            compareV1 = vec[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 17, 18], :]
            compareV2 = vec[[1, 2, 3, 7, 5, 6, 7, 9, 9, 10, 11, 13, 14, 15, 17, 18, 19, 19], :]
            #     compareV1 = vec[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17],:]
            #     compareV2 = vec[[1, 2, 3, 5, 6, 7 ,9, 10, 11, 13, 14, 15, 17, 18, 19],:]

            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angle = np.degrees(angle)
            return angle

        def joint_to_angle_alpha(landmark):

            # 21 x,y,z np array 생성
            joint = np.zeros((21, 3))

            for j, lm in enumerate(landmark.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            joint1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            joint2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]

            # 각 좌표의 벡터를 구함
            vec = joint1 - joint2
            vec = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]

            #     compareV1 = vec[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 17,18],:]
            #     compareV2 = vec[[1, 2, 3, 7, 5, 6, 7, 9, 9, 10, 11, 13, 14, 15, 17, 18, 19,19],:]
            compareV1 = vec[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = vec[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]

            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angle = np.degrees(angle)
            return angle

        TRAIN_DATA_URL = './dataset/output_digit.csv'

        file = np.genfromtxt(TRAIN_DATA_URL, delimiter=',')
        file = np.delete(file, 0, axis=1)
        file = np.delete(file, 0, axis=0)
        angleFile = file[1:, :-1]
        labelFile = file[1:, -1]

        angle = angleFile.astype(np.float32)
        label = labelFile.astype(np.float32)
        knn = cv2.ml.KNearest_create()
        knn.train(angle, cv2.ml.ROW_SAMPLE, label)

        # file_alpha = np.genfromtxt('./dataset/DataSet.txt', delimiter=',')
        # angleFile_alpha = file_alpha[:, :-1]
        # labelFile_alpha = file_alpha[:, -1]
        #
        # angle_alpha = angleFile_alpha.astype(np.float32)
        # label_alpha = labelFile_alpha.astype(np.float32)
        # knn_alpha = cv2.ml.KNearest_create()
        # knn_alpha.train(angle_alpha, cv2.ml.ROW_SAMPLE, label_alpha)

        with mp_hands.Hands(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            while cap.isOpened() and self.running:
                # print('while')
                success, image = cap.read()

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(imgRGB,
                                                  hand_landmarks,
                                                  mp_hands.HAND_CONNECTIONS)

                        input_data = joint_to_angle(hand_landmarks)
                        data = np.array([input_data], dtype=np.float32)
                        res, results, neighbours, dist = knn.findNearest(data, 3)

                        # 영어
                        if self.isEng:
                            index = chr(int(results[0][0]) + 65)
                            list = actions_alpha
                        # 숫자
                        else:
                            index = chr(int(results[0][0]) + 48)
                            list = actions

                        if index in list:
                            if len(sentence) > 0:
                                if index != sentence[-1]:
                                    if index != premotion:
                                        premotion = index
                                        start_time = time.time()
                                    else:
                                        # 시간이 1초가 넘어가면 삽입
                                        if time.time() - start_time > recognizeDelay:
                                            sentence.append(index)
                                            # if actions[np.argmax(res)] != sentence[-1]:
                                            # 이전 제스쳐와 다르면 시간재기
                                            # 문자 덧붙이기
                                            self.editText_me.insertPlainText(sentence[-1])
                            else:
                                sentence.append(index)
                                # 문자 덧붙이기
                                self.editText_me.insertPlainText(sentence[-1])

                # 화면 출력용 리사이즈
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # self.cam_me.resize(160, 120)

                h, w, c = imgRGB.shape
                qImg = QtGui.QImage(imgRGB.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
                self.cam_me.setPixmap(pixmap)

                # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, width)
                # cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)

                # convert Mat to byte for sending to client
                ret, imgencoded = cv2.imencode('.jpg', imgRGB)
                byte_img = np.array(imgencoded)

                int_len_byte_img = len(byte_img)
                bytes_len_img = int_len_byte_img.to_bytes(4, byteorder='big')

                # ba = QtCore.QByteArray()
                # buff = QtCore.QBuffer(ba)
                # buff.open(QtCore.QIODevice.WriteOnly)
                # ok = pixmap.save(buff, "png")
                # assert ok
                # pixmap_bytes = ba.data()
                # print(type(pixmap_bytes))

                if self.my_socket.socketType == video_cap.SERVER:
                    self.my_socket.targetSocket.send(bytes_len_img)
                    self.my_socket.targetSocket.send(byte_img)
                elif self.my_socket.socketType == video_cap.CLIENT:
                    self.my_socket.my_socket.send(bytes_len_img)
                    self.my_socket.my_socket.send(byte_img)
                else:
                    pass
        cap.release()
        cv2.destroyAllWindows()

    #################################################
    def receive_all(self, sock, count):
        buf = bytearray(b'')

        while count:
            newbuff = sock.recv(count)
            if not newbuff:
                return None
            buf += newbuff
            count -= len(newbuff)
        return buf

    def bytes_to_int(self, bytes):
        result = 0

        for b in bytes:
            result = result * 256 + int(b)

        return result

    def int_to_bytes(self, value, length):
        result = []

        for i in range(0, length):
            result.append(value >> (i * 8) & 0xff)

        result.reverse()

        return result

    def recv_thread(self):
        # 서버, 클라이언트 소켓 판별
        if self.my_socket.socketType == video_cap.SERVER:
            while True:
                # dataSize = self.my_socket.targetSocket.recv(4)
                # print(int.from_bytes(dataSize, byteorder='big'))
                # data = self.my_socket.targetSocket.recv(int.from_bytes(dataSize, byteorder='big'))
                #
                # ba = QtCore.QByteArray(data)
                # pixmap = QtGui.QPixmap()
                # ok = pixmap.loadFromData(ba, "png")
                # assert ok
                bytes_buf = self.receive_all(self.my_socket.targetSocket, 4)
                bytes_length = self.bytes_to_int(bytes_buf)
                print("Rx Length = {} ".format(bytes_length))
                byte_data = self.receive_all(self.my_socket.targetSocket, int(bytes_length))
                # convert jpg image to matix
                g_decode_img = np.frombuffer(byte_data, dtype=np.uint8)
                g_decode_img = cv2.imdecode(g_decode_img, cv2.COLOR_RGB2BGR)

                h, w, c = g_decode_img.shape
                qImg = QtGui.QImage(g_decode_img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                self.cam_you.setPixmap(pixmap)

        elif self.my_socket.socketType == video_cap.CLIENT:
            while True:
                # dataSize = self.my_socket.my_socket.recv(4)
                # print(int.from_bytes(dataSize, byteorder='big'))
                # data = self.my_socket.my_socket.recv(int.from_bytes(dataSize, byteorder='big'))
                #
                # ba = QtCore.QByteArray(data)
                # pixmap = QtGui.QPixmap()
                # ok = pixmap.loadFromData(ba, "png")
                # assert ok
                #
                # self.cam_you.setPixmap(pixmap)
                bytes_buf = self.receive_all(self.my_socket.my_socket, 4)
                bytes_length = self.bytes_to_int(bytes_buf)
                print("Rx Length = {} ".format(bytes_length))
                byte_data = self.receive_all(self.my_socket.my_socket, int(bytes_length))
                # convert jpg image to matix
                g_decode_img = np.frombuffer(byte_data, dtype=np.uint8)
                g_decode_img = cv2.imdecode(g_decode_img, cv2.COLOR_RGB2BGR)

                h, w, c = g_decode_img.shape
                qImg = QtGui.QImage(g_decode_img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)

                self.cam_you.setPixmap(pixmap)
        else:
            pass

    def make_thread(self):
        try:
            if self.th is None:
                print('make_thread before')
                self.th = threading.Thread(target=self.capStart)
                print('make_thread after')
        except AttributeError:
            print('AttributeError')
            pass

    def stop(self):
        print('stop (running state) : ', self.running)
        try:
            self.running = False
            self.th.join()
            self.th = None
            print('stop (thread joined)')
        except AttributeError:
            print('join pass')

    def start(self):
        print('start')
        self.running = True
        self.make_thread()
        print('start : call make_thread')
        self.th.start()
        print('start (th state) :', self.th)

    def on_exit(self):
        self.stop()

    def eng_num(self):
        if self.isEng:
            self.isEng = False
        else:
            self.isEng = True
