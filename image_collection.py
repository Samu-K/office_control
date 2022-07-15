# import opencv
import cv2 as cv

# import uuid
import uuid

# other imports
import os
import time

labels = ["stop", "two_up", "two_down","fist"]
number_imgs = 10

for label in labels:
    cap = cv.VideoCapture(0)
    print(f"Handling {label}")
    
    time.sleep(5)
    
    for img_num in range(number_imgs):
        print(f"Handling num {img_num} of 10")
        
        ret, frame = cap.read()
        frame = cv.resize(frame,(720,360))
        
        img_name = os.path.join(
            "workspace/images/collected_images",
            label,
            f"{label}.{img_num}.jpg")
        
        cv.imwrite(img_name,frame)
        cv.imshow('Frame',frame)
        
        time.sleep(2)
        
        if cv.waitKey(1) == ord("q"):
            break
        
cap.release()
cv.destroyAllWindows()