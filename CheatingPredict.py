from ultralytics import YOLO
import cv2 as cv 
import numpy as np
import torch

# Jarak konstant 
KNOWN_DISTANCE = 50 
PERSON_WIDTH = 10 
MOBILE_WIDTH = 3.0 

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)

FONTS = cv.FONT_HERSHEY_COMPLEX

model = YOLO('yolov8n.pt')
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

def object_detector(image):
    results = model(image)  
    data_list =[]
    for result in results:
        try:
            boxes = result.boxes
            box = boxes.xyxy[0]  
            classid = boxes.cls[0]
            score = boxes.conf[0]
        except:
            print("Cannot find class")
            continue

        box = box.int().tolist()
        print(box)
        classid = int(classid)
        
        color= COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        # cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid ==0: # person class id 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid ==67: # person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
    
    return data_list


def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance * 2.54


cap = cv.VideoCapture(0)
focal_person = 1467


isCheating = "Not Cheating"

while True:
    ret, frame = cap.read()

    data = object_detector(frame) 
    for d in data:
        x, y = d[2]
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] =='cell phone':
            x, y = d[2]

        if round(distance,2) > 70.00 or round(distance,2) <= 58.31 or d[0] =='cell phone':
            isCheating = "Cheating"
        else:
            isCheating = "Not Cheating"
        cv.rectangle(frame, (x, y-3), (x+150, y+23),BLACK,-1 )
        # cv.putText(frame, f'Dis: {round(distance,2)} cm', (x+5,y+13), FONTS, 0.48, GREEN, 2)
        cv.putText(frame, f'{isCheating}', (x+5,y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame',frame)
    
    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()