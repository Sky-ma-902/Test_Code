import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import cv2
import os

def class_photo(model,img_path,output_path):
    img = plt.imread(img_path)
    img1 = cv2.imread(img_path)
    results = model(img1)
    result_max = results[0].probs
    class_index = result_max.top1
    class_name  = results[0].names[class_index]
    class_confidence = result_max.top1conf.item() * 100

    text = f"{class_name}\n{class_confidence:.2f}%"
    cv2.putText(img1,text,(10,10),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.imshow("predict",img1)
    
    plt.imshow(img)
    plt.title(f"class:{class_name}\n confidence:{class_confidence:.2f}%")
    plt.axis('off')
    plt.show()

model = YOLO("yolo11n-cls.pt")
class_photo(model,"./bus.jpg","./gragh")

