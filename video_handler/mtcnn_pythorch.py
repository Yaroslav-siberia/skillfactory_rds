import os.path

import numpy as np
from facenet_pytorch import MTCNN
from image_match.goldberg import ImageSignature
import pickle
import time
import cv2
from yolov5 import YOLOv5
import torch
import torchvision.transforms as T
from PIL import Image

#import random

helmet_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "yolov5_m_hat.pt")
mask_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mask_classification_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


helmet_model = YOLOv5(helmet_model_path, device)  # сеть поиска касок
mtcnn = MTCNN(keep_all=True, device=device)  # сеть поиска лиц
mask_model = torch.load(mask_model_path,map_location=device)  # классификация лиц в маске или без
mask_model.eval()
gis = ImageSignature()


def add_new_face_frame(frame):
    """
    функция для добавления лиц по кадрам
    :param frame:
    :return:
    """
    _, width = frame.shape[1], frame.shape[0]
    k = 900 / width
    dim = (int(frame.shape[1] * k), int(frame.shape[0] * k))
    image = cv2.resize(frame, dim)
    return add_new_face_video(image)


def encode_face(face):
    """
    эта функция получает на вход лицо вырезанное из кадра.
    изменяет разрешение на 100*100 пикселей кодирует в вектор"""
    face_sig = None
    try:
        # face = cv2.resize(face, (100, 100))
        face_sig = gis.generate_signature(face)
    except cv2.error:
        ...
    finally:
        return face_sig


def add_new_face_photo(image):
    """
    функция по добавлению получению кодировки лица через фотографию
    :param image: путь к фотографии
    :return:
    """
    return add_new_face_frame(cv2.imread(image))


async def async_add_new_face_video(cap, loop):
    """
    функция по получению кодировки лица через видео
    тут не закрывается видеопоток, его сюда только передаем как аргумент, закрывать надо вне функции
    :param cap: cv2.VideoCapture(0), 0 - индекс камеры которая обрабатывает
    :param loop:
    :return:face_sig - закодированнное лицо
    """
    a = True  # бесконечный цикл т.к. неизвестно когда захватится лицо
    while (a):
        ret, image = cap.read()  # получаем булевую переменную о захвате и сам кадр из видеопотока
        if ret:  # проверяем что кадр захвачен корректно
            boxes, probs = mtcnn.detect(image)  # получили список с координатами лиц и список вероятностей что это лицо
            if any(probs):  # если лицо или чтото подобное есть в кадре
                for prob, box in zip(probs, boxes):
                    if prob < 0.90:  # если вероятность захвата лица достаточно высока
                        # print("Плохое качество фотографии, повторите попытку")
                        return False, np.empty([])
                    else:
                        # сюда попали если лицо четко распознано
                        box = [int(v) for v in box]  # координаты лица в инт
                        image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                        cv2.resize(image_new, (100, 100))  # приводим все к одному масштабу
                        face_sig = encode_face(image_new)  # функция кодирования лица , синхронная
                        # cv2.imwrite("./11.jpg", image_new)  # сохраняем лицо в файлик это для теста
                        a = False
                        return True, face_sig


# cap=cv2.VideoCapture(0)#объект видеопотока
# loop = asyncio.get_event_loop()# бесконечный цикл
# code = result = loop.run_until_complete(async_add_new_face_video(cap,loop))# цикл работает до тех пор пока не будет результат

def add_new_face_video(frame):
    """
    функция по получению кодировки лица через видео, синхронный вариант
    :param frame:
    :return: face_sig - закодированнное лицо
    """
    boxes, probs = mtcnn.detect(frame)  # получили список с координатами лиц и список вероятностей что это лицо
    if probs.any() and probs.size == 1:  # если лицо или чтото подобное есть в кадре
        for prob, box in zip(probs, boxes):
            if prob >= 0.90:  # если вероятность захвата лица достаточно высока
                box = [int(v) for v in box]  # координаты лица в инт
                image_new = frame[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                cv2.resize(image_new, (100, 100))  # приводим все к одному масштабу
                face_sig = encode_face(image_new)  # функция кодирования лица , синхронная
                # cv2.imwrite("./11.jpg", image_new)  # сохраняем лицо в файлик это для теста
                return frame[box[1]:box[3], box[0]:box[2]], face_sig
    return None, None


def test_arg(frame):
    """
    Просто функция чтобы показать что в класс передается функция вебхука
    :param frame:
    :return:
    """
    #a=random.randint(0,10000)
    #cv2.imwrite(str(a)+'.jpg', frame)  # сохраняем frame в файлик это для теста
    print(1)

# trans convert image to torch.tensor
trans = T.Compose([
        T.Resize(200),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class video_handler:
    """
    класс обработки видео кадра
    осуществляет поиск лиц
    определение свой/чужой


    главная функция - handle_frame.
        сначала ищутся лица с помощью search_faces
        если найдено 1 и более лицо проверяем на свой/чужой
            если найдены чужие то выделяем цветом.
            вот причина дернуть колбэк по причине постороннего лица в кадре
    """

    def __init__(self, faces_list, d_limit=0.62, p_limit=0.8, h_limit=0.7, detecting_items=None, allocating=None,
                 callback=None):
        """
        :param faces_list: список с кодировками лиц работников
        :param d_limit:  пороговое значение на детектирование лица
        :param p_limit: пороговое значение на степень схожести обнаруженного лица с списком кодировок
        :param detecting_items: список объектов для детектирования (каски маски) пока не определен строго
        :param callback: callback для кадров с неизвестными лицами
        """
        if detecting_items is None:  # по дефолту детектируем и проверяем все
            detecting_items = {'Face': True, 'Helmet': True, 'Mask': True}
        if allocating is None:  # по дефолту выделяем нарушение и не выделяем "разрешения"
            allocating = {'Alarms': True, 'Agreement': False}
        self.face_list = faces_list
        self.d_limit = d_limit
        self.p_limit = p_limit
        self.h_limit = h_limit
        self.detect_faces = detecting_items['Face']
        self.detecting_Helmet = detecting_items['Helmet']
        self.detecting_Mask = detecting_items['Mask']
        self.allocating_alarms = allocating['Alarms']
        self.allocating_agreement = allocating['Agreement']
        self.callback = callback
        self.helmet_model = helmet_model
        self.mask_model = mask_model
        # get class names
        self.names = self.helmet_model.model.names if hasattr(self.helmet_model, 'model') else self.helmet_model.names

    def update_face_list(self, face_list):
        """
        чтобы обновить наш список с закодированными лицами
        :param face_list:
        :return:
        """
        self.face_list = face_list

    def handle_frame(self, frame):
        """
        ищем лица в кадре
        проверяем лица с известными
        если есть неизвесмтные лица - аларм
        :param frame:
        :param func:
        :return:
        """
        # переменные показывающие по какому типу контроля произошло нарушение
        face_alarm = False
        helmet_alarm = False
        mask_alarm = False
        # списки в которые будут помещаться боксы с задетектированным нарушением
        face_alarm_boxes = []
        helmet_alarm_boxes = []
        mask_alarm_boxes = []
        # списки в которые будут помещаться боксы с задетектированным соблюдением правил
        face_agreed_boxes = []
        helmet_agreed_boxes = []
        mask_agreed_boxes = []
        # кадр в котором выделены нарушения и который отправляется в колбэк
        new_frame = None
        # цвета выделения
        color0 = (0, 0, 255)  # red
        color1 = (255, 0, 0)  #blue

        # поиск лиц обязателен если у нас идет проверка масок или проверка свой\чужой
        if self.detect_faces or self.detecting_Mask:
            face_boxes, detected_faces = self.search_faces(frame)
        # проверяем лица
        if self.detect_faces and len(detected_faces) >= 1:
            face_alarm, face_alarm_boxes, face_agreed_boxes = self.check_faces(detected_faces, face_boxes)
        # проверяем маски
        if self.detecting_Mask and len(detected_faces) >= 1:
            mask_alarm, mask_alarm_boxes, mask_agreed_boxes = self.face_mask_classification(detected_faces, face_boxes)
        # проверяем каски
        if self.detecting_Helmet:
            helmet_alarm, helmet_alarm_boxes, helmet_agreed_boxes = self.detect_helmet_heads(frame)
        # к этому моменту мы потенциально собрали все наши нарушения и "разрешения" и боксы для них
        # и выделяем
        if self.allocating_alarms:
            all_alarms = face_alarm_boxes + helmet_alarm_boxes + mask_alarm_boxes
            new_frame = self.allocating_boxes(frame, all_alarms, color0)
        if self.allocating_agreement:
            # разрешения выделяем на том же кадре на котором выделили нарушения
            all_agreement = face_agreed_boxes + helmet_agreed_boxes + mask_agreed_boxes
            new_frame = self.allocating_boxes(new_frame, all_agreement, color1)
        # если есть хоть какое-то нарушение то оповещаем
        if face_alarm or helmet_alarm or mask_alarm:
            self.callback(new_frame)





    # checked
    def allocating_boxes(self, frame, boxes_list, color):
        new_frame = frame.copy()
        for box in boxes_list:
            new_frame = cv2.rectangle(new_frame, (box[0], box[1]), (box[2], box[3]), color=color, thickness=2)
        return new_frame


    # checked
    def face_mask_classification(self,faces,boxes):
        mask_alarm = False

        box_without_mask = []
        box_with_mask = []
        for face,box in zip(faces,boxes):
            face = cv2.resize(face, (200, 200))
            face_new = Image.fromarray(face)
            tens = trans(face_new)
            tens = tens.unsqueeze(0)
            tens = tens.to(device)
            pred = self.mask_model(tens)
            answer = int(pred.argmax())

            # здесь применяется изменение координат бокса на +4 для того чтобы рамка проверки лица и
            # рамка проверки в маске ли лицо не наслаивались друг на друга
            if answer == 0:  #with mask
                box = [int(a)+4 for a in box]
                box_with_mask.append(box)
            else:  #wothout mask
                box = [int(a)+4 for a in box]
                box_without_mask.append(box)
                mask_alarm = True
        return mask_alarm, box_without_mask, box_with_mask

    # checked
    def detect_helmet_heads(self, frame):
        '''
        predictions classes:
        0 - head
        1 - helmet
        :param frame:
        :return:
        '''
        with_helmet = []
        without_helmet = []
        helmet_alarm = False
        size = max(list(frame.shape))
        results = self.helmet_model.predict(frame, size=size)
        predictions = results.pred[0]
        if len(predictions) > 0:  # если обнаружили что-то из наших категорий то выделяем
            boxes = predictions[:, :4]
            scores = predictions[:, 4].tolist()
            categories = predictions[:, 5].tolist()
            for category, box, score in zip(categories, boxes,scores):
                if category == 0 and score > self.h_limit:
                    box = [int(a) for a in box]
                    without_helmet.append(box)
                    helmet_alarm = True
                if category == 1 and score > self.h_limit:
                    box = [int(a) for a in box]
                    with_helmet.append(box)
        return helmet_alarm, without_helmet, with_helmet

    # checked
    def search_faces(self, frame):
        """
        ищем лица в кадре
        :param frame: обрабатываемый кадр
        :return: agreed_boxes список рамок лиц
        """
        detected_faces = []
        agreed_boxes = []
        boxes, probs = mtcnn.detect(frame)
        if any(probs):  #
            for prob, box in zip(probs, boxes):
                if prob >= self.p_limit:
                    # механизм может найти лицо которое частично в кадре и предположить его рамку за кадром...
                    box = [0 if v < 0 else int(v) for v in box]
                    face = frame[box[1]:box[3], box[0]:box[2]]
                    detected_faces.append(face)
                    agreed_boxes.append(box)
        return agreed_boxes, detected_faces

    # checked
    def check_faces(self, faces, boxes):
        """
        проверяем список полученных лиц с базой и если обнаруживаем неизвестное лицо - выделяем его в кадре
        :param faces: список вырезанных лиц
        :param boxes: границы для каждого лица
        :return:
        """
        alien_boxes = []  # неопознанные лица
        mine_boxes = []  # опознанные лица
        detected_alien = False
        for face, box in zip(faces, boxes):
            face_sig = encode_face(face)
            check_list = 0
            for code in self.face_list:
                distance = gis.normalized_distance(face_sig, code)
                if distance < self.d_limit:
                    # уже нашли похожее лицо, нечего по списку дальше идти
                    mine_boxes.append(box)
                    break
                else:
                    check_list += 1
            if check_list == len(self.face_list):  # по сути не вышли из прошлого цикла и ни с кем не совпало
                detected_alien = True
                alien_boxes.append(box)
        return detected_alien, alien_boxes, mine_boxes



# небольшой набросок кода для тестов
# в пикле список одировок лиц
# экземпляр класса специально по кадрам дергается
# нового человека, тебя нет в сериализованном списке) детект точно будет
# test_arg функция для вебкуха и тд)ну как бы ее псевдоним в тесте
#


def testing_video_handler_class():
    with open('data.pickle', 'rb') as pickle_in:
        face_list = pickle.load(pickle_in)
    print(len(face_list))
    d_limit = 0.6
    p_limit = 0.8
    h_limit = 0.7
    detecting_items = {'Face': True, 'Helmet': True, 'Mask': True}
    allocating = {'Alarms': True, 'Agreement': True}
    vh = video_handler(face_list, d_limit, p_limit,h_limit, detecting_items, allocating, test_arg)
    cap = cv2.VideoCapture('/home/ysiberia/123/kask_detection2.mp4')
    frames_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            vh.handle_frame(frame)
            frames_count += 1
            #if time.time() > finish:
            #    break
    #print('fps = ', frames_count / 60)


if __name__ == '__main__':
    testing_video_handler_class()
