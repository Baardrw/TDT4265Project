import torchmetrics.detection
import torchmetrics.detection.mean_ap
from datamodule import CityscapesDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
import torchmetrics

from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import munch
import yaml
from pathlib import Path

torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Setting up the valid labels
        
        self.num_classes = len(VALID_LABELS)
        
        # Setting up model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if config.use_pretrained_weights else None
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features, num_classes=self.num_classes)
        
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
                "val/loss": losses
            }
        ,on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))        
    
    def test_step(self, batch, batch_idx):
        images, targets = batch
        
        self.model.eval()
        outputs = self.model(images)
        self.val_map.update(outputs, targets)
        map = self.val_map.compute()
        
        self.log_dict(
            {
                "test/map_50": map['map_50'],
                "test/map_75": map['map_75']
            }
        ,on_epoch=True, on_step=False, prog_bar=True, sync_dist=True, batch_size=len(images))
        

if __name__ == "__main__":
    
    pl.seed_everything(42)
    
    dm = CityscapesDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_root=config.data_root,
        mode=config.mode,
        valid_labels=VALID_LABELS,
        label2idx=STR2IDX,
    )
    
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
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
            EarlyStopping(monitor="val/map_50", patience=config.early_stopping_patience, mode="max", verbose=True),
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
