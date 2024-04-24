from ultralytics.engine.trainer import BaseTrainer

from datamodule import CityscapesDataModule
from munch import munchify

import torch
from datamodule import CityscapesDataModule
import torch
from torch import nn
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path
import ultralytics
from ultralytics.nn.tasks import DetectionModel


config = munch.munchify(yaml.load(open("YOLO/config.yaml"), Loader=yaml.FullLoader))

VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

class CustomTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def get_dataset(self):
        pass
        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "train/loss": loss,
            "train/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        loss = self.loss_fn(y_hat, y)
        self.log_dict({
            "val/loss":loss,
            "val/acc": acc
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        acc = self.acc_fn(y_hat, y)
        self.log_dict({
            "test/acc": acc,
        },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)



if __name__ == "__main__":


    # dm = CityscapesDataModule(        
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     data_root=config.data_root,
    #     mode=config.mode,
    #     valid_labels=VALID_LABELS,
    #     label2idx=STR2IDX,
    #     image_dimensions=[config.image_h, config.image_w],
    #     )
    
    
    # trainer = CustomTrainer(config = "YOLO/modelconfig.yaml")
    # model = DetectionModel()
    # trainer.model = model
    # trainer.train_loader = dm.train_dataloader()



    # trainer.train()

    model = YOLO('yolov8n.yaml')