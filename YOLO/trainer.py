from datamodule import CityscapesDataModule
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path
import ultralytics
from ultralytics.models import YOLO
from ultralytics.engine.trainer import BaseTrainer

torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("YOLO/config.yaml"), Loader=yaml.FullLoader))

VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.num_classes = len(VALID_LABELS)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.model.fc = nn.Linear(2048, self.config.num_classes)
    
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
    
    pl.seed_everything(42)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    dm = CityscapesDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        data_root=config.data_root,
        mode=config.mode,
        valid_labels=VALID_LABELS,
        label2idx=STR2IDX,
        image_dimensions=[config.image_h, config.image_w],
    )

    # dm = CIFAR100DataModule(
    #     batch_size=config.batch_size,
    #     num_workers=config.num_workers,
    #     train_split_ratio=config.train_split_ratio,
    #     data_root=config.data_root
    # )
    # if config.checkpoint_path:
    #     model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    #     print("Loading weights from checkpoint...")
    # else:
    #     model = LitModel(config)

    trainer = BaseTrainer(

        # cfg = config.yaml,
        # check_val_every_n_epoch=config.check_val_every_n_epoch,
        # enable_progress_bar=config.enable_progress_bar,
        # precision="bf16-mixed",
        # deterministic=True,
        # logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
        # callbacks=[
        #     EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
        #     LearningRateMonitor(logging_interval="step"),
        #     ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
        #                     filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
        #                     auto_insert_metric_name=False,
        #                     save_weights_only=True,
        #                     save_top_k=1),
        # ]
        )
    if not config.test_model:
        trainer.fit(model, datamodule=dm)
    
    trainer.test(model, datamodule=dm)
