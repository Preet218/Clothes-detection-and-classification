import detectron2

import numpy as np
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer

## Encoding the categories
l = ['botas','monos','camisas','zapatos','corbatas','sombreros','bufandas','pantalones','vestidos','abrigos','cinturones'] ## Using only 11 classes which has instances in the dataset
dic = {}
n = len(l)
count = 0
for i in l:
    dic[i] = count
    if count >= n:
        count = 0
    count += 1


## Converting the metadata in json file to detectron2 compactible COCO format
def get_balloon_dicts(img_dir):
    global dic

    dataset_dicts = []
    for filename in os.listdir(img_dir):
        if filename[-4:] != 'json':
            continue
        json_file = os.path.join(img_dir, filename)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        
        filename = os.path.join(img_dir, imgs_anns["file_name"])
        for idx, v in enumerate(imgs_anns.values()):
            if v[-3:] == 'jpg':
                continue
            record = {}

            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            objs = []
            for i in v:
                try:
                    px = [j for j in range(int(i['x']), int(i['x']+i['width']),5)]
                    py = [j for j in range(int(i['y']), int(i['y']+i['height']),5)]
                    poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                    poly = [p for x in poly for p in x]
                    obj = {
                        "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": dic[i['class']],
                    }
                except:
                    continue

                if obj["category_id"] < 0 or obj["category_id"] > n:
                    print("This is the fault", obj["category_id"])
                    continue
                objs.append(obj)
            record["annotations"] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts

## Registering the dataset
for d in ["train"]:
    DatasetCatalog.register("train" + d, lambda d=d: get_balloon_dicts("train/"))
    MetadataCatalog.get("train" + d).set(thing_classes=['botas','monos','camisas','zapatos','corbatas','sombreros','bufandas','pantalones','vestidos','abrigos','cinturones'])
balloon_metadata = MetadataCatalog.get("traintrain")

## Getting the name of the classes 
class_names = MetadataCatalog.get("traintrain").thing_classes

## for checking the COCO format
dataset_dicts = get_balloon_dicts("train")

# Checking if the created COCO format is correct
for d in random.sample(dataset_dicts, 5):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("annotated image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)

## Setting up the model configuration 
# Trained the model for a total of 1000 iterations
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("traintrain",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
## This is after the model was trained for 100 iterations
model_path = os.path.join('output', "model_final.pth")
cfg.MODEL.WEIGHTS = model_path
cfg.SOLVER.IMS_PER_BATCH = 2 
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 1000   
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 

## Ready for training and saving the model
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()