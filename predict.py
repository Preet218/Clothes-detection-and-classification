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

## Setting up the configurations
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("traintrain",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2  
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 100    
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11 

## Importing model weights
model_path = os.path.join('output', "model_final.pth")
cfg.MODEL.WEIGHTS = model_path

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

## Prediction classes
class_names = ['botas','monos','camisas','zapatos','corbatas','sombreros','bufandas','pantalones','vestidos','abrigos','cinturones']

class Metadata:
    def get(self, _):
        return class_names

## Predicting and Visualizing the results
for d in os.listdir("pict"):
    img_path = os.path.join('pict', d)   
    im = cv2.imread(img_path)
    outputs = predictor(im) 
    v = Visualizer(im[:, :, ::-1], Metadata, scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Predicted Image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)