# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:14:30 2020

@author: Asus
"""

import cv2
import numpy as np


img=cv2.imread("C:/YOLO1/custom_yolo_model/yolov4/darknet/CARPK_data/car_images/20161225_TPZ_00132.png")   

img_width=img.shape[1]
img_height=img.shape[0]

img_blob=cv2.dnn.blobFromImage(img, 1/255, (416,416), swapRB=True, crop=False)

labels = ["car","empty"]
colors = ["255,0,0","0,255,255"]
colors = [np.array(color.split(",")).astype("int") for color in colors]
colors = np.array(colors)


model = cv2.dnn.readNetFromDarknet("C:/YOLO1/custom_yolo_model/yolov4/darknet/CARPK_yolov4.cfg",
                                   "C:/YOLO1/custom_yolo_model/yolov4/CARPK_weights/backup/CARPK_yolov4_last.weights")
layers = model.getLayerNames()
output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]

model.setInput(img_blob)
detection_layers = model.forward(output_layer)

ids_list=[]
boxes_list=[]
confidence_list=[]


for detection_layer in detection_layers:
    for object_detection in detection_layer:
        
        scores = object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.15:
            
            label = labels[predicted_id]
            bounding_box = object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_width/2))
            start_y = int(box_center_y - (box_height/2))
            
            ids_list.append(predicted_id)
            confidence_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

i=0
j=0
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.50, 0.4)
for max_id in max_ids:
    max_class_id = max_id[0]
    box=boxes_list[max_class_id]
    
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]
    
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidence_list[max_class_id]
            
            
    end_x = start_x + box_width
    end_y = start_y + box_height
    
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]
    
    if label == "car":
        i=i+1
    else:
        j=j+1
        
    label = "{}: {:.2f}%".format(label, confidence*100)
    
      
    print("Predicted object {}".format(label))
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
   
    
with open("C:/YOLO1/custom_yolo_model/yolov4/darknet/CARPK_data/car_labels/20161225_TPZ_00132.txt", "r") as f:
    a=f.read().count("\n1")
    
  
with open("C:/YOLO1/custom_yolo_model/yolov4/darknet/CARPK_data/car_labels/20161225_TPZ_00132.txt", "r") as f:
    b=f.read().count("\n0")

print("\nBOŞ PARK YERİ SAYISI:",a)
print("ARAÇ SAYISI:",b)
print("\nTESPİT EDİLEN BOS PARK YERİ SAYISI={}".format(j))   
print("TESPİT EDİLEN ARAÇ SAYISI={}".format(i))



c = (1-abs(i-b)/b)*100
d = (1-abs(j-a)/a)*100

print("\nARAÇ TESPİTİ BAŞARI ORANI: %", c)
print("BOŞ PARK ALANI TESPİT BAŞARI ORANI: %", d)


cv2.namedWindow("Detection Window",cv2.WINDOW_NORMAL)
cv2.imshow("Detection Window", img)