
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



"""This file holds helper functions for creating and training Faster R-CNN models"""


FOCAL_LOSS_WEIGHTS = [1.0, 191.6178861788618, 44.30263157894737, 22.816069699903196, 40.08333333333333, 2.770867622854456, 12.496818663838813, 2.162095220621961]
FOCAL_LOSS_WEIGHTS = torch.tensor(FOCAL_LOSS_WEIGHTS)
GAMMA = 2.0

def set_model_transform(model):
    coco_mean = [0.485, 0.456, 0.406]
    coco_std = [0.229, 0.224, 0.225]
    
    coco_mean_grayscale = [0.2989 * coco_mean[0] + 0.5870 * coco_mean[1] + 0.1140 * coco_mean[2]]# The exact formula used by torchvision.transforms.Grayscale
    coco_std_grayscale = [math.sqrt(0.2989**2 * coco_std[0]**2 + 0.5870**2 * coco_std[1]**2 + 0.1140**2 * coco_std[2]**2)]     
    print(coco_mean_grayscale, coco_std_grayscale)
    
    transform = GeneralizedRCNNTransform(
                                        min_size=800,
                                        max_size=1333,
                                        image_mean=coco_mean_grayscale,
                                        image_std=coco_std_grayscale
                                        )
    
    model.transform = transform
    
def faster_rcnn_focal_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.
    
    Modified from the pytorch fastrcnn loss implementation to use focal loss instead of cross entropy.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)
    
    pt = torch.exp(-classification_loss)
    
    # Calculate focal weights for each label
    flw = FOCAL_LOSS_WEIGHTS.to(labels.device)
    focal_weights = torch.gather(flw, 0, labels)

    # Compute focal loss for each sample in the batch
    focal_loss = focal_weights * (1 - pt) ** GAMMA * classification_loss

    # Calculate the mean focal loss over the batch
    loss = focal_loss.mean()
        
    print(f"Classification loss: {classification_loss}, Focal loss: {loss}")

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

    
def init_faster_rcnn(config, litmodel):
    if config.use_focal_loss:
        roi_heads.fastrcnn_loss = faster_rcnn_focal_loss
    
        litmodel.num_classes = len(VALID_LABELS)
        
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
