import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#  for webcam

# cap = cv2.VideoCapture(0)# it will work on 0 nor one since i have only one web cam that can be default one
# cap.set(3,1280)
# cap.set(4,720)

#  for video

cap = cv2.VideoCapture("road.mp4")


model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("mask.jpg")

tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.3)

limits=[0,750,3000,750]

totalcounts=[]


while True:
    success,img = cap.read()



    imgregion=cv2.bitwise_and(img,mask)

    imggraphic=cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imggraphic,(0,0))

    res=model(imgregion,stream=True)
    detections=np.empty((0,5))
    for r in res:
        boxes=r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            w, h = x2-x1,y2-y1
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            curr=classNames[cls]

            if curr == 'car' or curr == 'motorbike' or curr == 'bus' or curr == 'truck' and conf>0.3:
                # cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1,offset=3)
                # cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=5)
                currArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currArray))

    resultstracker=tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for resulte in resultstracker:
        x1,y1,x2,y2,id = resulte
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(resulte)
        cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=7)

        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[3]+20:
            if totalcounts.count(id)==0:
                totalcounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'count : {len(totalcounts)}', (50,50))
    cv2.putText(img,str(len(totalcounts)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
    cv2.imshow("image",img)
    # cv2.imshow('imgregion',imgregion)
    cv2.waitKey(1)

#