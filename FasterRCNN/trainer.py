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

import munch
import yaml
from pathlib import Path

from nl_datamodule import NapLabDataModule
from cs_datamodule import CityscapesDataModule


torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

VALID_LABELS = ['background', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'car']

if not config.pre_train:
    VALID_LABELS = [
            "background",
            "truck",
            "bus",
            "bicycle",
            "scooter",
            "person",
            "rider",
            "car"
        ]
    
STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

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

    

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Setting up focal loss, super hacky but works
        if config.use_focal_loss:
            roi_heads.fastrcnn_loss = faster_rcnn_focal_loss
    
        self.num_classes = len(VALID_LABELS)
        
        # Setting up model

        if config.use_pretrained_weights:
            self.model = fasterrcnn_resnet50_fpn(weights=None)
            
            # Loading weights finetuned on COCO greyscale https://huggingface.co/Theem/fasterrcnn_resnet50_fpn_grayscale
            # Load pretrained weights
            state_dict = torch.load(self.config.pretrained_weights_path)['model']
            # Adapt input convolution
            self.model.backbone.body.conv1 = torch.nn.Conv2d(1, 64,
                                        kernel_size=(7, 7), stride=(2, 2),
                                        padding=(3, 3), bias=False).requires_grad_(True)
            
            
            
            self.model.load_state_dict(state_dict)
            
            
            # Changing the transforms to grayscale
            set_model_transform(self.model)
            
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=self.num_classes)
            
           
            
            # TODO: figure out appropriate anchor generator
            # # Create custom anchor generator
            # anchor_generator = AnchorGenerator(
            #     sizes=((32,), (64,), (128,), (256,), (512,)),
            #     aspect_ratios=tuple([(0.05, 0.1, 0.25, 0.5 ,1.0, 2.0) for _ in range(5)]))
            
            # self.model.rpn.anchor_generator = anchor_generator
            
            # self.model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        else:
            # TODO:
            NotImplementedError("No support for training from scratch yet.")         
       
        # self.model.box_detections_per_img = 256 # TODO: Perhaps not so good maybe is good doe, idkk
        # Setting up the metric
        self.val_map = torchmetrics.detection.mean_ap.MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox", 
        )
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, image):
        """Inference step."""
        self.model.eval()
        output = self.model(image)
        
        return output

    def training_step(self, batch, batch_idx):
        
        # if self.current_epoch == 5:
        #     self.model.backbone.trainable_layers = 0
        
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log_dict({
            "train/loss": losses,
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))
        return losses
    
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        
        self.model.train()
        loss_dict = self.model(images, targets)        
        losses = sum(loss for loss in loss_dict.values())
        
        self.model.eval()
        outputs = self.model(images)
    
        self.val_map.update(outputs, targets)
        map = self.val_map.compute()
        
        self.log_dict(
            {
                "val/map_50": map['map_50'],
                "val/map_75": map['map_75'],
                "val/map": map['map'],
                "val/map_small": map['map_small'],
                "val/map_medium": map['map_medium'],
                "val/map_large": map['map_large'],
                
                "val/loss": losses
            }
        ,on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))        
    
    def test_step(self, batch, batch_idx):
        images = [i.unsqueeze(0) for i in  batch[0] ]
        
        self.model.eval()
        outputs = self.model(images) 
        
        # self.val_map.update(outputs, targets)
        # map = self.val_map.compute()
        
        # self.log_dict(
        #     {
        #         "test/map_50": map['map_50'],
        #         "test/map_75": map['map_75'],
                
        #         "test/map": map['map'],
        #         "test/map_small": map['map_small'],
        #         "test/map_medium": map['map_medium'],
        #         "test/map_large": map['map_large'],
        #     }
        # ,on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))
        
        self.draw_boxes(images, outputs)
        

    def draw_boxes(self, images, outputs):
        mean = [0.3090844516698354]
        std = [0.17752945677448584]
        
        mean = [0.4701497724237717] # taken from data analysis of naplab dataset
        std = [0.2970671789647479]
        
        for i in range(len(images)):
            output = outputs[i]
            
            print(output['boxes'].shape)
            
            keep = ops.nms(output['boxes'], output['scores'], iou_threshold=0.3)
            output['boxes'] = torch.index_select(output['boxes'], 0, keep)
            output['labels'] = torch.index_select(output['labels'], 0, keep)
            
            print(output['boxes'].shape)
            
            image_tensor = v2.functional.to_dtype(images[i], torch.float32)
            de_normed = add(mul(image_tensor, self.coco_std_grayscale[0]), self.coco_mean_grayscale[0])
            de_normed = add(mul(image_tensor, std[0]), mean[0])
            
            image_uint8 = (de_normed * 255).type(torch.uint8)
            # print(output['boxes'])
            
            labels = [VALID_LABELS[label_id] for label_id in output['labels']]
            img = utils.draw_bounding_boxes(image_uint8, output['boxes'], labels=labels, width=1) # TODO: labels

            img = img.cpu()
            import matplotlib.pyplot as plt
            
            if img.numpy().transpose(1, 2, 0).shape[2] == 1:
                plt.imsave(f"inferences/test{i}.png", img.numpy().transpose(1, 2, 0)[:, :, 0], cmap='gray')
            else:
                plt.imsave(f"inferences/test{i}.png", img.numpy().transpose(1, 2, 0))            
        
        exit()
            

def prog_res():
    """Only for fine tuning on the naplab dataset the aim is to  s.t. it doesnt overfit instantly"""
    sizes = [16, 32, 64, 128, 256, 512]
    checkpoint_folder = f"/work/baardrw/checkpoints/nap/{config.wandb_project}/{config.wandb_experiment_name}/"
    
    
    for i, size in enumerate(sizes):
        
        if i < config.progressive_resizing_start_index:
            continue
        
        dm = NapLabDataModule(
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                data_root=config.data_root,
                image_dimensions=[config.image_h, config.image_w],
                resize_dims=[size, size]
            )
        

        if i > 0:
            model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_folder + f"prog_res{i - 1}.ckpt", config=config)
            print("Loading weights from checkpoint...") 

        else:
            model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
            
            
        trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/map", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename=f'prog_res{i}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
        
        
        trainer.fit(model, datamodule=dm)
        
           

if __name__ == "__main__":
    
    pl.seed_everything(42)
    
    
    if config.progressive_resizing:
        print("Progressive resize fine tuning")
        prog_res()
        exit()
    
    if config.pre_train:
        dm = CityscapesDataModule(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_root=config.data_root,
            mode=config.mode,
            valid_labels=VALID_LABELS,
            label2idx=STR2IDX,
            image_dimensions=[config.image_h, config.image_w],
        )
    else:
        dm = NapLabDataModule(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_root=config.data_root,
            image_dimensions=[config.image_h, config.image_w],
        )
    
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
        
    
        # for param in model.model.backbone.parameters():
        #     print(param.requires_grad)
        
        # print("\n\n")
        # for param in model.model.rpn.parameters():
        #     print(param.requires_grad)
        # print("\n\n")
            
        # for param in model.model.roi_heads.parameters():
        #     print(param.requires_grad)
        
        # if not config.pre_train:
        #     # model.model.box_detections_per_img = 53            
        #     pass        
        
        # print("HWLOOOOO")
            

    else:
        model = LitModel(config)
        
        
    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.max_epochs, 
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        enable_progress_bar=config.enable_progress_bar,
        precision="bf16-mixed",
        # deterministic=True,
        logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        callbacks=[
            EarlyStopping(monitor="val/map", patience=config.early_stopping_patience, mode="max", verbose=True),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
                            filename='best_model:epoch={epoch:02d}-val_map_50={val/map_50:.4f}',
                            auto_insert_metric_name=False,
                            save_weights_only=True,
                            save_top_k=1),
        ])
    
    
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
