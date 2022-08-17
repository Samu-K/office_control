# imports
from cProfile import label
import cv2 as cv
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util, visualization_utils, config_util
from object_detection.builders import model_builder


# load pipeline config to build model
configs = config_util.get_configs_from_pipeline_file("tensorflow/workspace/models/cmod/pipeline.config")
model = model_builder.build(model_config=configs["model"],is_training=False)

# ckpt
ckpt = tf.compat.v2.train.Checkpoint(model=model)

ckpt.restore("tensorflow/workspace/models/cmod/ckpt-4").expect_partial()


# setup function for detection
def detect(image):
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image,shapes)
    
    return model.postprocess(prediction_dict, shapes)

# Setup category index
category_index = label_map_util.create_category_index_from_labelmap("tensorflow/workspace/annotations/label_map.pbtxt")

cap = cv.VideoCapture(0)

# setup loop to go through frames
while cap.isOpened():
    # read frame
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np,0),dtype=tf.float32)
    detections = detect(input_tensor)
    
    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0,:num_detections].numpy() for key, value in detections.items()}
    
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    
    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"]+label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=2,
        min_score_thresh=.8,
        agnostic_mode=False
    )
    
    cv.imshow("object_detection", cv.resize(image_np_with_detections,(800,600)))
    
    wk = cv.waitKey(1)
    if wk & 0xFF == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        break
    
    