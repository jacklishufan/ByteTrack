from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import cv2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("audiogram_segmentation_train",
#                         #"audiogram_original"
# )
# cfg.DATASETS.TEST = ("audiogram_segmentation_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE='cpu'
# cfg.SOLVER.IMS_PER_BATCH = 2
# cfg.SOLVER.BASE_LR = 1e-4
# single_iteration = 1* cfg.SOLVER.IMS_PER_BATCH
# iterations_for_one_epoch = int(369 / single_iteration)
# cfg.SOLVER.MAX_ITER = iterations_for_one_epoch * 20
# cfg.TEST.EVAL_PERIOD = iterations_for_one_epoch * 5
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
# cfg.TEST.EVAL_PERIOD = 100



cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well


cfg.OUTPUT_DIR = "output_segmentation_new"
predictor = DefaultPredictor(cfg)
test_input = cv2.imread('videos/fish.jpg')
r = predictor(test_input)
#print(r)