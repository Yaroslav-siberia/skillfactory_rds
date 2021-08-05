import cv2
from facenet_pytorch import MTCNN
import os
import glob
mtcnn = MTCNN(keep_all=True, device="cpu")



ind=0
'''
for filename in glob.iglob('RMFD/*/*.jpg', recursive=True):
    image = cv2.imread(filename)
    try:
        boxes, probs = mtcnn.detect(image)
        if any(probs):  # если лицо или чтото подобное есть в кадре
            for prob, box in zip(probs, boxes):
                if prob > 0.90:  # если вероятность захвата лица достаточно высока
                    # сюда попали если лицо четко распознано
                    box = [int(v) for v in box]  # координаты лица в инт
                    image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                    try:
                        image_new = cv2.resize(image_new, (200, 200))
                        cv2.imwrite(f'./face/with_mask/{ind}.jpg', image_new)
                        ind += 1
                        print(ind)
                    except Exception as e:
                        print(e)
                    
    except:
        continue
'''
for filename in glob.iglob('with_mask/*.jpg', recursive=True):
    image = cv2.imread(filename)
    try:
        boxes, probs = mtcnn.detect(image)
        if any(probs):  # если лицо или чтото подобное есть в кадре
            for prob, box in zip(probs, boxes):
                if prob > 0.90:  # если вероятность захвата лица достаточно высока
                    # сюда попали если лицо четко распознано
                    box = [int(v) for v in box]  # координаты лица в инт
                    image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                    try:
                        image_new = cv2.resize(image_new, (200, 200))
                        cv2.imwrite(f'./face/with_mask/{ind}.jpg', image_new)
                        ind += 1
                        print(ind)
                    except Exception as e:
                        print(e)
                    
    except:
        continue

print('111111111111111111111111111111111111111111111111')

'''
ind=0
for filename in glob.iglob('without_mask/*.jpg', recursive=True):
    image = cv2.imread(filename)
    try:
        boxes, probs = mtcnn.detect(image)
        if any(probs):  # если лицо или чтото подобное есть в кадре
            for prob, box in zip(probs, boxes):
                if prob > 0.90:  # если вероятность захвата лица достаточно высока
                    # сюда попали если лицо четко распознано
                    box = [int(v) for v in box]  # координаты лица в инт
                    image_new = image[box[1]:box[3], box[0]:box[2]]  # вырезаем лицо из фотки
                    try:
                        image_new = cv2.resize(image_new, (200, 200))
                        cv2.imwrite(f'./face/without_mask/{ind}.jpg', image_new)
                        ind += 1
                        print(ind)
                    except Exception as e:
                        print(e)

    except:
        continue
'''
