# import opencv
import cv2 as cv

# import uuid
import uuid

# other imports
import os
import time

labels = ["fist"]
number_imgs = 20
cap = cv.VideoCapture(0)

print("Starting up")
ret, frame = cap.read()
cv.imshow("Frame",frame)
fp = "tensorflow/workspace/images/collected_images"

for label in labels:
    label_path = os.path.join(fp,label)
    if os.path.exists(label_path) == False:
        os.mkdir(label_path)
    
    print(f"Handling {label}")
    for img_num in range(number_imgs):
        print(f"Handling num {img_num+1} of 10")
        
        print("Taking on 3")
        for num in range(1,4):
            print(num)
            time.sleep(1)
        ret, frame = cap.read()
        img_name = os.path.join(
            "tensorflow/workspace/images/collected_images",
            label,
            f"{label}_{img_num}.jpg")
        
        cv.imwrite(img_name,frame)
        cv.imshow('Frame',frame)
        
        time.sleep(2)
        
        if cv.waitKey(1) == ord("q"):
            break
        
cap.release()
cv.destroyAllWindows()