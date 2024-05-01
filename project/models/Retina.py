import math
import torchmetrics.detection
import torchmetrics.detection.mean_ap
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import torchmetrics

from torch import nn, mul, add
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator, RPNHead
from torchvision.models.detection import roi_heads

from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.transforms import v2
from torchvision import utils, ops
import torch.nn.functional as F

from functools import partial


def set_model_transform(model, litmodel):
    coco_mean = [0.485, 0.456, 0.406]
    coco_std = [0.229, 0.224, 0.225]
    
    coco_mean_grayscale = [0.2989 * coco_mean[0] + 0.5870 * coco_mean[1] + 0.1140 * coco_mean[2]]# The exact formula used by torchvision.transforms.Grayscale
    coco_std_grayscale = [math.sqrt(0.2989**2 * coco_std[0]**2 + 0.5870**2 * coco_std[1]**2 + 0.1140**2 * coco_std[2]**2)]     
    print(coco_mean_grayscale, coco_std_grayscale)
    
    litmodel.coco_mean_grayscale = coco_mean_grayscale
    litmodel.coco_std_grayscale = coco_std_grayscale
    transform = GeneralizedRCNNTransform(
                                        min_size=800,
                                        max_size=1333,
                                        image_mean=coco_mean_grayscale,
                                        image_std=coco_std_grayscale
                                        )
    
    model.transform = transform
  
def init_retina(config, litmodel):


    # custom_ag = AnchorGenerator(
    #     sizes=((32, 64, 128, 256, 512),),
    #     aspect_ratios=((0.5, 1.0, 2.0),),
    # )
    litmodel.model = retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT)
    # fastmodel = fasterrcnn_resnet50_fpn(weights=None)
        
    # # Loading weights finetuned on COCO greyscale https://huggingface.co/Theem/fasterrcnn_resnet50_fpn_grayscale
    # # Load pretrained weights
    # state_dict = torch.load(litmodel.config.pretrained_weights_path)['model']
    # # Adapt input convolution
    # fastmodel.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
    #                             kernel_size=(7, 7), stride=(2, 2),
    #                             padding=(3, 3), bias=False).requires_grad_(True)
    
    # fastmodel.load_state_dict(state_dict)

    # Adapt input convolution
    litmodel.model.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
                                kernel_size=(7, 7), stride=(2, 2),
                                padding=(3, 3), bias=False).requires_grad_(True)

    # Adapt detection head to use 8 (+ 1) classes
    num_anchors = litmodel.model.head.classification_head.num_anchors


    litmodel.model.head.classification_head = RetinaNetClassificationHead(
        in_channels = litmodel.model.backbone.out_channels,
        num_classes = 8, 
        num_anchors = num_anchors,
        norm_layer = partial(torch.nn.GroupNorm, 32)
    )




    
    # Changing the transforms to grayscale
    set_model_transform(litmodel.model, litmodel)
    # TODO: figure out appropriate anchor generator
        
