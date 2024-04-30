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

from datasets.nl_datamodule import NapLabDataModule
from datasets.cs_datamodule import CityscapesDataModule
# from custom_models.FasterRCNN import init_faster_rcnn



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


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = len(VALID_LABELS)        
        
        if config.model == 'faster_rcnn':
            init_faster_rcnn(config, self)
        if config.model == 'yolo':
            print('loading yolo model')
            # model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m') 

        
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
        print(image.shape)
        output = self.model.forward(image)
        
        return output

    def training_step(self, batch, batch_idx):
        
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
    checkpoint_folder = f"/work/ianma/checkpoints/nap/{config.wandb_project}/{config.wandb_experiment_name}/"
    
    
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
    
    # if config.checkpoint_path:
    #     model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    #     print("Loading weights from checkpoint...")
        
    #     # Freeze backbone
            
    #     for param in model.model.backbone.parameters():
    #         print(param.requires_grad)
        
    #     if not config.pre_train: # TODO: ?  
    #         # model.model.box_detections_per_img = 53            
    #         pass        
        

    # else:
    #     model = LitModel(config)
        
        
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
    model = LitModel(config)

    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
