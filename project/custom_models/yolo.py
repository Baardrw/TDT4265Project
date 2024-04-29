
import math
import torchmetrics.detection
import torchmetrics.detection.mean_ap
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import torchmetrics

from torch import nn, mul, add
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator, RPNHead
from torchvision.models.detection import roi_heads

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2
from torchvision import utils, ops
import torch.nn.functional as F



def init_yolo(config, litmodel):


    if config.use_focal_loss:
        roi_heads.fastrcnn_loss = faster_rcnn_focal_loss
        
        # Setting up model

        if config.use_pretrained_weights:
            litmodel.model = fasterrcnn_resnet50_fpn(weights=None)
            
            # Loading weights finetuned on COCO greyscale https://huggingface.co/Theem/fasterrcnn_resnet50_fpn_grayscale
            # Load pretrained weights
            state_dict = torch.load(litmodel.config.pretrained_weights_path)['model']
            # Adapt input convolution
            litmodel.model.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
                                        kernel_size=(7, 7), stride=(2, 2),
                                        padding=(3, 3), bias=False).requires_grad_(True)
            
            
            
            litmodel.model.load_state_dict(state_dict)
            
            # Changing the transforms to grayscale
            set_model_transform(litmodel.model)
            
            in_features = litmodel.model.roi_heads.box_predictor.cls_score.in_features
            litmodel.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=litmodel.num_classes)
            
           
            
            # TODO: figure out appropriate anchor generator
            
            # Create custom anchor generator
            # anchor_generator = AnchorGenerator(
            #     sizes=((32,), (64,), (128,), (256,), (512,)),
            #     aspect_ratios=tuple([(0.2, 0.35, 0.5, 1.0, 2.0) for _ in range(5)]))
            
            # litmodel.model.rpn.anchor_generator = anchor_generator
            
            # litmodel.model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
            
        else:
            # TODO:
            NotImplementedError("No support for training from scratch yet, also never will be.")         
       
        # litmodel.model.box_detections_per_img = 53
