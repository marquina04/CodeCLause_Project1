import cv2


img = cv2.imread('object.jpeg')

cap=cv2.VideoCapture(0)
cap.set(3,648)
cap.set(4,488)

classNames =[]
classfile='./coco.names'
with open('Object_Detection_Files\coco.names','rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

config='Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weights='Object_Detection_Files/frozen_inference_graph.pb'

net=cv2.dnn_DetectionModel(weights,config)
net.setInputSize(328,328)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
    success,img=cap.read()
    classids, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classids, bbox)

    if len(classids)!=0:
        for classid,conf,box in zip(classids.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box, color=(255,0,0),thickness=2)
            cv2.putText(img,classNames[classid-1].upper(),(box[0]+10,box[1]+38), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    cv2.imshow('image',img)
    cv2.waitKey(1)