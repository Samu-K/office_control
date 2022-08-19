# imports
from cProfile import label
from re import A
import cv2 as cv
import numpy as np
import tensorflow as tf
import os
import time
from object_detection.utils import label_map_util, visualization_utils, config_util
from object_detection.builders import model_builder
import subprocess
from datetime import datetime
from lamp_control import Lamp


# load pipeline config to build model
configs = config_util.get_configs_from_pipeline_file("tensorflow/workspace/models/cmod/pipeline.config")
model = model_builder.build(model_config=configs["model"],is_training=False)

# ckpt
ckpt = tf.compat.v2.train.Checkpoint(model=model)

ckpt.restore("tensorflow/workspace/models/cmod/ckpt-7").expect_partial()

# setup function for detection
def detect(image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image,shapes)
    
    return model.postprocess(prediction_dict, shapes)

# Setup category index
category_index = label_map_util.create_category_index_from_labelmap("tensorflow/workspace/annotations/label_map.pbtxt")

cap = cv.VideoCapture(0)
away_counter = 0
stop_counter = 0
fist_counter = 0
stopped = False
lamp_state = False
label_id_offset = 1

def det_brightness(current_time):
    x = current_time.hour + (current_time.minute / 60)
    level = 0.0142*x**4 - 0.8115*x**3 + 15.523*x**2 - 111.57*x + 283.29
    return level

with open("hue_key.txt","r") as f:
    hue = dict()
    for line in f:
        spl = line.strip().split(":")
        hue[spl[0]] = spl[1]
    f.close()

office_lamp = Lamp(hue["office_id"],hue["bridge_ip"],hue["username"])

office_lamp.power(state=True)
office_lamp.brightness(det_brightness(datetime.now().time()))

# setup loop to go through frames
while cap.isOpened():
    away_frame = False
    
    # read frame
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np,0),dtype=tf.float32)
    detections = detect(input_tensor)
    
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}
    
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    image_np_with_detections = image_np.copy()
    
    all_detections = []
    for index, value in enumerate(detections["detection_classes"]+label_id_offset):
        if detections["detection_scores"][index] > 0.8:
            all_detections.append(category_index[value]["name"])
        
    if "stop" in all_detections:
        stop_counter += 1
    
    if "fist" in all_detections:
        fist_counter += 1
        
    if not stopped:
        if ("Samu" or "Isse") not in all_detections:
            away_counter += 1
            away_frame = True
    
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"]+label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.7,
        agnostic_mode=False
    )
    
    cv.imshow("object_detection", cv.resize(image_np_with_detections,(800,600)))

    if away_frame == False:
        pass
        away_counter = 0
    elif away_counter >= 15:
        print("Shutting down")
        cap.release()
        cv.destroyAllWindows()
        os.system("psshutdown64.exe -d -t 2")
        
    if stop_counter >= 3:
        if not stopped:
            stopped = True
            print("Program stopped")
        else:
            stopped = False
            print("Program continued")
        time.sleep(1)
        stop_counter = 0
    
    if fist_counter >= 3:
        if lamp_state == True:
            office_lamp.power(False)
            lamp_state = False
        else:
            office_lamp.power(True)
            lamp_state = True
        fist_counter = 0
        
    wk = cv.waitKey(1)
    if wk & 0xFF == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        office_lamp.power(False)
        break