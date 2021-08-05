from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
import cv2
from image_match.goldberg import ImageSignature
import pickle

resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=True, device='cpu')
gis = ImageSignature()


def add_new_face(image):
    image = cv2.imread(image)

    boxes, probs = mtcnn.detect(image)  # получили список с координатами лиц и список вероятностей что это лицо
    print(probs)
    if probs:
        for prob, box in zip(probs, boxes):
            if prob < 0.50:
                print("Плохое качество фотографии, повторите попытку")
            else:
                box = [int(v) for v in box]
                image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                cv2.resize(image_new, (100, 100))
                cv2.imshow('face', image_new)
                face_sig = encode_face(image_new)  # кодируем лицо
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                return face_sig
    else:
        print("Плохое качество фотографии, повторите попытку")


def encode_face(face):
    ''''эта функция получает на вход лицо вырезанное из кадра. изменяет разрешение на 100*100 пикселей
    кодирует в вектор'''
    face_sig = gis.generate_signature(face)
    return face_sig


def photo_compare():
    face0 = add_new_face("./test1/1.jpg")  # фронтальная без всего
    face1 = add_new_face("./test1/2.jpg")  # профиль слева без всего
    face2 = add_new_face("./test1/3.jpg")  # профиль справа без всего
    face3 = add_new_face("./test1/4.jpg")  # фронтальная в маске
    face4 = add_new_face("./test1/5.jpg")  # профиль слева в маске
    face5 = add_new_face("./test1/6.jpg")  # профиль справа в маске
    face6 = add_new_face("./test1/7.jpg")  # фронтальная в каске
    face7 = add_new_face("./test1/8.jpg")  # профиль слева в каске
    face8 = add_new_face("./test1/9.jpg")  # профиль справа в каске
    face9 = add_new_face("./test1/10.jpg")  # фронтальная в каске и маске
    face10 = add_new_face("./test1/11.jpg")  # профиль слева в каске и маске
    face11 = add_new_face("./test1/12.jpg")  # профиль справа в каске и маске
    face12 = add_new_face("./test1/13.jpg")  # я в очках
    face13 = add_new_face("./test1/14.jpg")  # другой человек
    face14 = add_new_face("./test1/15.jpg")  # другой человек d rfcrt b vfcrt
    faces = [face0, face1, face2, face3, face4, face5, face6, face7, face8, face9, face10, face11, face12, face13,
             face14]
    answers = {}
    print(len(faces))
    for i in range(0, len(faces)):
        for j in range(i, len(faces)):
            a = gis.normalized_distance(faces[i], faces[j])
            comment = "дистанция между face {0} face {1}".format(i, j)
            answers[comment] = a
    with open('data.pickle', 'wb') as f:
        pickle.dump(faces, f)
    for k in answers.keys():
        print(k, answers[k])


def cap_face():
    cap = cv2.VideoCapture(0)
    a = True
    while (a):
        ret, image = cap.read()
        if ret:
            boxes, probs = mtcnn.detect(image)  # получили список с координатами лиц и список вероятностей что это лицо
            cv2.imshow('image', image)
            print(probs)
            if probs:
                for prob, box in zip(probs, boxes):
                    if prob < 0.90:
                        print("Плохое качество фотографии, повторите попытку")
                    else:
                        cv2.imwrite("./test1/11.jpg", image)
                        print("captured")
                        cap.release()
                        cv2.destroyAllWindows()
                        a = False
                        break


def write_captured_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    while (True):
        ret, image = cap.read()
        if ret:
            boxes, probs = mtcnn.detect(image)  # получили список с координатами лиц и список вероятностей что это
            if probs[0] is not None:
                if len(probs) >= 1:
                    for prob, box in zip(probs, boxes):
                        if prob >= 0.60:
                            box = [int(v) for v in box]
                            image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)
        cv2.imshow("capturing face", image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


async def add_new_face_video(cap, loop):
    '''
    :param cap: cv2.VideoCapture(0), 0 - индекс камеры которая обрабатывает
    :param loop:
    :return:face_sig - закодированнное лицо
    '''
    a = True  # бесконечный цикл т.к. неизвестно когда захватится лицо
    while (a):
        ret, image = cap.read()  # получаем булевую переменную о захвате и сам кадр из видеопотока
        if ret:  # проверяем что кадр захвачен корректно
            boxes, probs = mtcnn.detect(image)  # получили список с координатами лиц и список вероятностей что это лицо
            if probs:  # если лицо или чтото подобное есть в кадре
                for prob, box in zip(probs, boxes):
                    if prob < 0.90:  # если вероятность захвата лица достаточно высока
                        print("Плохое качество фотографии, повторите попытку")
                    else:
                        # сюда попали если лицо четко распознано
                        box = [int(v) for v in box]  # координаты лица в инт
                        image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                        cv2.resize(image_new, (100, 100))  # приводим все к одному масштабу
                        face_sig = encode_face(image_new)  # функция кодирования лица , синхронная
                        cv2.imwrite("./11.jpg", image_new)  # сохраняем лицо в файлик это для теста
                        a = False
                        return face_sig


# cap = cv2.VideoCapture(0)  # объект видеопотока
# loop = asyncio.get_event_loop()  # бесконечный цикл
# code = result = loop.run_until_complete(
#     add_new_face_video(cap, loop))  # цикл работает до тех пор пока не будет результат
# print(code)
