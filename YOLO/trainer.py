import sys
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import Accuracy
import munch
import yaml
from pathlib import Path
from ultralytics.models import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from datamodule import CityscapesDataModule


torch.set_float32_matmul_precision('medium')
config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

# class LitModel(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config

#         weights = ResNet50_Weights.DEFAULT if config.use_pretrained_weights else None
#         self.model = resnet50(weights=weights)
#         self.model.fc = nn.Linear(2048, self.config.num_classes)
        
#         self.loss_fn = nn.CrossEntropyLoss()
#         self.acc_fn = Accuracy(task="multiclass", num_classes=self.config.num_classes)
    
#     def configure_optimizers(self):
#         optimizer = torch.optim.SGD(self.parameters(), lr=self.config.max_lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
#         lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.max_epochs)
#         return [optimizer], [{"scheduler": lr_scheduler, "interval": "epoch"} ]

#     def forward(self, x):
#         y_hat = self.model(x)
#         return y_hat

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         loss = self.loss_fn(y_hat, y)
#         acc = self.acc_fn(y_hat, y)
#         self.log_dict({
#             "train/loss": loss,
#             "train/acc": acc
#         },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         acc = self.acc_fn(y_hat, y)
#         loss = self.loss_fn(y_hat, y)
#         self.log_dict({
#             "val/loss":loss,
#             "val/acc": acc
#         },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
    
#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.forward(x)
#         acc = self.acc_fn(y_hat, y)
#         self.log_dict({
#             "test/acc": acc,
#         },on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

class CustomTrainer(DetectionTrainer):
    def __init__(self, dm):
        self.datamodule = dm
        self.args = get_cfg('yolo.yaml', overrides)

    
    def get_dataloader(self):
        return self.datamodule



if __name__ == "__main__":
    
    VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
    STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

    dm = CityscapesDataModule(
        data_root='/work/ianma/cityscapes',
        valid_labels=VALID_LABELS,
        label2idx=STR2IDX,
        image_dimensions=[256, 512]
    )
    
    
    if config.checkpoint_path:
        model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
        print("Loading weights from checkpoint...")
    else:
        model = YOLO()

    # trainer = pl.Trainer(
    #     devices=config.devices, 
    #     max_epochs=config.max_epochs, 
    #     check_val_every_n_epoch=config.check_val_every_n_epoch,
    #     enable_progress_bar=config.enable_progress_bar,
    #     precision="bf16-mixed",
    #     # deterministic=True,
    #     logger=WandbLogger(project=config.wandb_project, name=config.wandb_experiment_name, config=config),
    #     callbacks=[
    #         EarlyStopping(monitor="val/acc", patience=config.early_stopping_patience, mode="max", verbose=True),
    #         LearningRateMonitor(logging_interval="step"),
    #         ModelCheckpoint(dirpath=Path(config.checkpoint_folder, config.wandb_project, config.wandb_experiment_name), 
    #                         filename='best_model:epoch={epoch:02d}-val_acc={val/acc:.4f}',
    #                         auto_insert_metric_name=False,
    #                         save_weights_only=True,
    #                         save_top_k=1),
    #     ])
    trainer = CustomTrainer(dm)
    
    trainer.train()
