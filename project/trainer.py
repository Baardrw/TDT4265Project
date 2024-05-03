import math
from typing import List
import numpy as np
import torchmetrics.detection
import torchmetrics.detection.mean_ap
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import torchmetrics

from torch import nn, mul, add
from torchvision.models.detection.faster_rcnn import AnchorGenerator, RPNHead

from torchvision.transforms import v2
from torchvision import utils, ops

import munch
import yaml
from pathlib import Path

from datasets.nl_datamodule import NapLabDataModule
from datasets.cs_datamodule import CityscapesDataModule
from models.FasterRCNN import init_faster_rcnn
from models.Retina import init_retina


# SAHI stuff
from sahi.prediction import ObjectPrediction


torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

VALID_LABELS = ['background', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'car'] # Labels that are valid for the cityscapes dataset

if not config.pre_train:
    VALID_LABELS = ["background", "truck", "bus", "scooter", "bicycle", "person", "rider", "car"] # Labels that are valid for the naplab dataset
    
STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.stage = 0
        self.config = config
        self.num_classes = len(VALID_LABELS)   
        
        self.object_prediction_list: List[ObjectPrediction]= [] # Object predictions list used for sahi
        
        self.mean = [0.5085652989351241] # taken from data analysis of naplab dataset
        self.std = [0.2970477123406435]     
        
        if config.model == 'faster_rcnn':
            print("Using Faster RCNN")
            init_faster_rcnn(config, self)
        else:
            init_retina(config, self)
        
        self.val_map = torchmetrics.detection.mean_ap.MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox", 
            # max_detection_thresholds=[1, 6, 12],
        )
    
    
    def configure_optimizers(self):
        max_lr = self.config.max_lr
        momentum = self.config.momentum
        weight_decay = self.config.weight_decay
        max_epochs = self.config.max_epochs
        
        if config.staged_training:
            if self.stage == 0:
                max_lr = self.config.head_max_lr
                max_epochs = self.config.head_max_epochs
            else:
                max_lr = self.config.all_max_lr
                max_epochs = self.config.all_max_epochs
        
        optimizer = torch.optim.SGD(self.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]
    

    def forward(self, image):
        """Inference step."""
        self.model.eval()
        output = self.model(image)
        
        return output

    def training_step(self, batch, _):
        
        images, targets = batch
        
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) 
        
        self.log_dict({
            "train/loss": losses,
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))
        return losses
    
        
    def validation_step(self, batch, _):
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
        
    
    def test_step(self, _, __):
        pass
    
        # Model is tested using sahi in sahi_demo.py
        
        
    def perform_inference(self, image: np.ndarray):
        """
        Args:
            -image: image ad contigous numpy array
        """
        
        def naplab_preprocess(image):
            formatted_ndarray = np.moveaxis(image, -1, 0)
            image_tensor = torch.tensor(formatted_ndarray)
            
            
            transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(num_output_channels=1),
                v2.RandomEqualize(p=1.0),
                v2.Resize((512, 512)),
                v2.Normalize(mean=self.mean, std=self.std),
            ])
            
            
            return transforms(image_tensor)

        image = naplab_preprocess(image)
        image = image.to(self.device)
        self.model.eval()
        outputs = self.model([image])
        
        # The bounding boxes of the outputs are now only valid for the 512x512 image
        # We need to convert them to the original image size 128x256
        # y is scaled by 4 and x is scaled by 2 
                
        outputs[0]['boxes'][:, 0] *= 0.5 # minx
        outputs[0]['boxes'][:, 1] *= 0.25 # miny 
        outputs[0]['boxes'][:, 2] *= 0.5
        outputs[0]['boxes'][:, 3] *= 0.25
    

        self.sahi_outputs = outputs
        
        # Normalize to 
        
    def convert_original_predictions(self, shift_amount, full_shape) -> ObjectPrediction:
        # print(shift_amount)
        # print(full_shape)
        # print(self.sahi_outputs)k
        
        for prediction in self.sahi_outputs:
            boxes = prediction['boxes'].cpu().detach()
            labels = prediction['labels'].cpu().detach()
            scores = prediction['scores'].cpu().detach()

            boxes[:, 0] += shift_amount[0]
            boxes[:, 1] += shift_amount[1]
            boxes[:, 2] += shift_amount[0]
            boxes[:, 3] += shift_amount[1]
            
            
            if full_shape is not None:
                boxes[:, 0] = torch.clamp(boxes[:, 0], 0, full_shape[1])
                boxes[:, 1] = torch.clamp(boxes[:, 1], 0, full_shape[0])
                boxes[:, 2] = torch.clamp(boxes[:, 2], 0, full_shape[1])
                boxes[:, 3] = torch.clamp(boxes[:, 3], 0, full_shape[0])
            
            for i in range(len(boxes)):
                if scores[i] > 0.4:
                    self.object_prediction_list.append(
                        ObjectPrediction(
                            category_id=int(labels[i]),
                            category_name=VALID_LABELS[labels[i]],
                            score=scores[i],
                            bbox=[boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                        )
                    )

    def reset_sahi_detections(self):
        self.object_prediction_list = []
    
    
    
def staged_traing():
    """Only fine tune the model head, then fine tune the entire model. Helps to prevent overfitting."""
    
    dm = NapLabDataModule(
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            data_root=config.data_root,
            image_dimensions=[config.image_h, config.image_w],
        )
    
    model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    
    if config.use_custom_anchor_gen:
        print("Using custom anchor generator")
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=tuple([(0.2, 0.35, 0.5, 1.0, 2.0) for _ in range(5)]))
        
        # Change anchor generator in RetinaNet if needed
        
    
    model.stage = 0
   
    ##  backbone:
    for param in model.model.backbone.parameters():
        param.requires_grad = False
    # Freeze head
    for param in model.model.head.classification_head.parameters():
        param.requires_grad = True
    
    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.head_max_epochs, 
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
                            save_top_k=1,
                            monitor="val/map",
                            mode="max"              # SAVE only the best model 
                            ),
        ])
        
        
        
    trainer.fit(model, datamodule=dm)
    
    model.stage = 1
     ## UnFreeze backbone:
    for i, param in enumerate(model.model.backbone.parameters()):
        if i > 22:
            param.requires_grad = True

    for param in model.model.head.classification_head.parameters():
        param.requires_grad = True
    
   
    trainer = pl.Trainer(
        devices=config.devices, 
        max_epochs=config.all_max_epochs, 
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
                            save_top_k=1,
                            monitor="val/map",
                            mode="max"              # SAVE only the best model 
                            ),
                            
        ])
    
        
    trainer.fit(model, datamodule=dm)
    
    

def prog_res():
    """Only for fine tuning on the naplab dataset the aim is to  s.t. it doesnt overfit instantly"""
    sizes = [16, 32, 64, 128, 256, 512]
    
    name = config.wandb_experiment_name
    
    for i, size in enumerate(sizes):
        
        if i < config.progressive_resizing_start_index:
            print("skipped size", size)
            continue
        
        dm = NapLabDataModule(
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                data_root=config.data_root,
                image_dimensions=[config.image_h, config.image_w],
                resize_dims=[size, size]
            )
        
        if size == 256:
            config.max_epochs = 10
            config.max_lr = 0.0005
            
        elif size == 512:
            config.max_epochs = 10
            config.max_lr = 0.00005

        if i > 0:
            checkpoint_folder = f"/work/baardrw/checkpoints/{config.wandb_project}/{config.wandb_experiment_name}/"
            model = LitModel.load_from_checkpoint(checkpoint_path=checkpoint_folder + f"prog_res{i - 1}.ckpt", config=config)
            print("Loading weights from checkpoint...") 

        else:
            model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        
        
        config.wandb_experiment_name = f"{config.wandb_experiment_name.split('@')[0]}@{size}"
        
        
        
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
                            save_top_k=1,
                            monitor="val/map",
                            mode="max"              # SAVE only the best model 
                            ),
                            
        ])
        
        
        trainer.fit(model, datamodule=dm)
        
           

if __name__ == "__main__":
    
    pl.seed_everything(42)
    
    
    if config.progressive_resizing:
        print("Progressive resize fine tuning")
        prog_res()
        exit()
        
    elif config.staged_training:
        print("Staged training")
        staged_traing()
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
        
        # Freeze backbone
            
        for param in model.model.backbone.parameters():
            print(param.requires_grad)
        
        if not config.pre_train: # TODO: ?  
            # model.model.box_detections_per_img = 53            
            pass        
        

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
    

    for param in model.model.backbone.parameters():
      print(param.requires_grad)
    
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    # figure out how many are kept from rpn 
    
    trainer.test(model, datamodule=dm)
