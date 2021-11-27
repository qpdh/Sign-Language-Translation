import threading
import mediapipe as mp
import cv2
import time
import numpy as np
from PyQt5 import QtGui

global running
global isEng
running = False
isEng = False

def capStart(qtLabel, textEdit_me):
    global running
    global isEng

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    recognizeDelay = 0.5
    start_time = time.time()
    premotion = 0

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    sentence = []
    actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    actions_alpha = [chr(i) for i in range(65, 91)]
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
        while cap.isOpened() and running:
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
                    if isEng:
                        index = chr(int(results[0][0]) + 48)
                        list = actions
                    # 숫자
                    else:
                        index = chr(int(results[0][0]) + 65)
                        list = actions_alpha

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
                                        textEdit_me.insertPlainText(sentence[len(sentence) - 1])
                        else:
                            sentence.append(index)
                            # 문자 덧붙이기
                            textEdit_me.insertPlainText(sentence[len(sentence) - 1])



            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            qtLabel.resize(width, height)
            h, w, c = imgRGB.shape
            qImg = QtGui.QImage(imgRGB.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg)
            qtLabel.setPixmap(pixmap)

            # 종료 조건
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
#################################################

#쓰레드 함수
global th
th = None

def make_thread(qtLabel, textEdit_me):
    global th
    try:
        if not th.isAlive():
            th = threading.Thread(target=capStart(qtLabel, textEdit_me))
    except AttributeError:
        th = threading.Thread(target=capStart(qtLabel, textEdit_me))

def stop(textEdit_me):
    global running
    global th
    running = False
    textEdit_me.clear()

def start(qtLabel, textEdit_me):
    global running
    global th
    running = True
    if threading.active_count() == 1:
        make_thread(qtLabel, textEdit_me)
        th.start()
    
def on_exit():
    stop()

def eng_num():
    global isEng
    global th
    if isEng:
        isEng = False
    else:
        isEng = True

def write_below(textEdit_me):

    print('a')