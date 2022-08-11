# import opencv
import cv2 as cv

# import uuid
import uuid

# other imports
import os
import time

labels = ["stop", "two_up", "two_down","fist"]
number_imgs = 10
cap = cv.VideoCapture(0)

print("Starting up")
ret, frame = cap.read()
frame = cv.resize(frame,(1080,720))
cv.imshow("Frame",frame)

for label in labels:
    print(f"Handling {label}")
    for img_num in range(number_imgs):
        print(f"Handling num {img_num+1} of 10")
        
        print("Taking on 5")
        for num in range(1,6):
            print(num)
            time.sleep(1)
        
        ret, frame = cap.read()
        frame = cv.resize(frame,(1080,720))
        
        img_name = os.path.join(
            "tensorflow/workspace/images/collected_images",
            label,
            f"{label}.{img_num}.jpg")
        
        cv.imwrite(img_name,frame)
        cv.imshow('Frame',frame)
        
        time.sleep(2)
        
        if cv.waitKey(1) == ord("q"):
            break
        
cap.release()
cv.destroyAllWindows()