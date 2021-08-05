import torch
import torchvision
import cv2
from facenet_pytorch import MTCNN
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import cm
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_class = torch.load('best_model.pth',map_location=device)
mask_class.eval()


cap = cv2.VideoCapture(0)
mtcnn = MTCNN(keep_all=True, device="cpu")
ind=0

trans = T.Compose([
T.Resize(200),
T.ToTensor(),
T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

while(True):
    ret, frame = cap.read()
    if ret:
        boxes, probs = mtcnn.detect(frame)
        if any(probs):
            print(1)
            for prob, box in zip(probs, boxes):
                if prob > 0.8:
                    box = [0 if v < 0 else int(v) for v in box]
                    face = frame[box[1]:box[3], box[0]:box[2]]
                    face = cv2.resize(face,(200,200))
                    #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_new = Image.fromarray(face)
                    tens=trans(face_new)
                    tens = tens.unsqueeze(0)
                    pred = mask_class(tens)
                    a=int(pred.argmax())
                    if a==0:
                        cv2.imwrite(f'./detected_with_mask/{ind}.jpg',face)
                        ind+=1
                    else:
                        cv2.imwrite(f'./detected_without_mask/{ind}.jpg', face)
                        ind += 1

cap.release()
cv2.destroyAllWindows()

