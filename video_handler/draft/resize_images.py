import cv2
import os
ind=0
import glob
for filename in glob.iglob('images/*.jpg', recursive=True):
    image = cv2.imread(filename)
    image_new = cv2.resize(image_new, (640, 640))
    cv2.imwrite(f'./hats/{ind}.jpg', image_new)
    ind+=1