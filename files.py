# imports
import os

custom_model = "cmod"
pre_model = "centernet_mobilenetv2_fpn_od"
url = "http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz"
tf_record_script_name = "generate_tfrecord.py"
label_map_name = "label_map.txt"

paths = {
    "workspace_path": os.path.join("workspace"),
    "scripts_path": os.path.join("scripts"),
    "api_path": os.path.join("models"),
    "annotation_path": os.path.join("workspace","annotations"),
    "image_path": os.path.join( "workspace", "images"),
    "model_path": os.path.join("workspace","models"),
    "pre_model_path": os.path.join("workspace","pre_trained_models"),
    "checkpoint_path": os.path.join("workspace","models",custom_model),
    "output_path": os.path.join( "workspace", "models",custom_model,"export"),
    "tfjs_path": os.path.join("workspace","models",custom_model,"tfjsexport"),
    "protoc_path": os.path.join("protoc")  
}

files = {
    "pipeline_config": os.path.join("workspace","models",custom_model,"pipeline.config"),
    "tf_record_script": os.path.join(paths["scripts_path"],tf_record_script_name),
    "labelmap": os.path.join(paths["annotation_path"],label_map_name)
}

for path in paths.values():
    if not os.path.exists(path):
        os.mkdir(path)
