# import object detection libraries
from concurrent.futures import thread
import cv2 as cv
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils, config_util
from object_detection.builders import model_builder

# import in additional libraries
import numpy as np
import os
import time
from datetime import datetime
from win10toast import ToastNotifier

# import in custom class for lamp control
from lamp_control import Lamp

# setup functions
def detect(image, model):
    """
    Detects objects withing the given image

    Args:
        image (tensor): TF tensor of frame or image to detect on
        model (tf model): Model used for detection
    Returns:
        Postprocessed prediction
    """
    # Preprocess our image
    image, shapes = model.preprocess(image)
    
    # Predict
    prediction_dict = model.predict(image,shapes)
    
    # Return the processed prediction
    return model.postprocess(prediction_dict, shapes)

def det_brightness(current_time) -> float:
    """
    Determines brightness percentage of lamp given the time of day

    Args:
        current_time (datetime object): The current time containing atleast hour and minute

    Returns:
        float : number indicating brightness level
    """
    # Fetch the current hour and minute
    # Sum them together
    x = current_time.hour + (current_time.minute / 60)
    
    # Use a function to map the time to a appropriate level of brightness
    level = 0.0142*x**4 - 0.8115*x**3 + 15.523*x**2 - 111.57*x + 283.29
    return level

# Setup variables
away_counter = 0
stop_counter = 0
fist_counter = 0
label_id_offset = 1
shut_down = False
stopped = False
lamp_state = False

# setup notification
toast = ToastNotifier()

# show notification of program starting
toast.show_toast(
    "Detection",
    f"Program started",
    duration=5,
    icon_path="icon.ico",
    threaded=True
)

# load pipeline config and build model
configs = config_util.get_configs_from_pipeline_file("tensorflow/workspace/models/cmod/pipeline.config")
model = model_builder.build(model_config=configs["model"],is_training=False)

# fetch checkpoint for trained model
ckpt = tf.compat.v2.train.Checkpoint(model=model)
ckpt.restore("tensorflow/workspace/models/cmod/ckpt-7").expect_partial()

# setup category index
category_index = label_map_util.create_category_index_from_labelmap("tensorflow/workspace/annotations/label_map.pbtxt")

# we read all the info needed to interact with Hue
with open("hue_key.txt","r") as f:
    hue = dict()
    for line in f:
        spl = line.strip().split(":")
        hue[spl[0]] = spl[1]
    f.close()

# setup our lamp object
office_lamp = Lamp(hue["office_id"],hue["bridge_ip"],hue["username"])

# turn on the lamp and set brightness
office_lamp.power(state=True)
office_lamp.brightness(det_brightness(datetime.now().time()))

# establish the capture device
cap = cv.VideoCapture(0)

# setup loop to go through frames
while cap.isOpened():
    # away detection is reset at beginning of loop
    away_frame = False
    # same for others
    stop_frame = False
    fist_frame = False
    
    # read frame
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    # conver our image to tensor and feed to detection function
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np,0),dtype=tf.float32)
    detections = detect(input_tensor,model)
    
    # check how many detections in frame
    num_detections = int(detections.pop("num_detections"))
    # read all detections to dict
    detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}
    
    # convert classe to int64
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    
    # setup a list to contain all detected classes
    all_detections = []
    
    # loop through classes
    for index, value in enumerate(detections["detection_classes"]+label_id_offset):
        # if they have confidence score over 80% we add them to the list
        if detections["detection_scores"][index] > 0.8:
            all_detections.append(category_index[value]["name"])
    
    # check to see if stop is detected
    if "stop" in all_detections:
        stop_counter += 1
        stop_frame = True
    
    if "fist" in all_detections:
        fist_counter += 1
        fist_frame = True
    
    # make sure presence detection is not stopped
    if not stopped:
        # see if anyone is present in room
        if ("Samu" or "Isse") not in all_detections:
            # if not we add to counter
            away_counter += 1
            # mak current frame absent
            away_frame = True

    image_np_with_detections = image_np.copy()
        
    # visualize detections
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"]+label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False
    )
    
    # show the final image
    cv.imshow("object_detection", cv.resize(image_np_with_detections,(800,600)))
    
    # check if current frame absent
    if away_frame == False:
        # if so we reset the counter
        away_counter = 0
    elif away_counter >= 15:
        # if no one has been detected for long enough
        # we set the computer to shut down and end detection loop
        shut_down = True
        break
    
    if stop_frame:
        # check to see if stop has been held for long enough
        if stop_counter >= 3:
            # flip the state of stopped
            if not stopped:
                stopped = True
                print("Program stopped")
                # show notification
                toast.show_toast(
                    "Detection",
                    f"Program paused",
                    duration=5,
                    icon_path="icon.ico",
                    threaded=True
                )

            else:
                stopped = False
                print("Program continued")
                # show notification
                toast.show_toast(
                    "Detection",
                    f"Program continued",
                    duration=5,
                    icon_path="icon.ico",
                    threaded=True
                )
                
            # reset our counter
            time.sleep(1)
            stop_counter = 0
    else:
        stop_counter=0
    
    if fist_frame:
        if fist_counter >= 6:
            # stop the program fully
            print("Stopping execution")
            # show notification
            toast.show_toast(
                "Detection",
                f"Program stopped",
                duration=5,
                icon_path="icon.ico",
                threaded=True
            )
            break
    else:
        fist_counter = 0

    # setup waitkey for opencv
    wk = cv.waitKey(1)
    
    # if waitkey detected break loop
    if wk & 0xFF == ord("q"):
        break

# once loop is done
# we destroy all windows and release camera
cv.destroyAllWindows()
cap.release()

# is shut down is commanded
if shut_down:
    # turn off lamp
    office_lamp.power(False)
    # set computer to sleep
    os.system("psshutdown64.exe -d -t 0")