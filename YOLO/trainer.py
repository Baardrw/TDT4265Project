import torch
from ultralytics.models import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

torch.set_float32_matmul_precision('medium')
# config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

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

# class CustomTrainer(DetectionTrainer):



if __name__ == "__main__":
    
    model = YOLO('yolov8s.yaml')
    
    # Datasets
    cityscapes = '/work/ianma/cityscapes_yolo/data.yaml'
    naplab = '/work/ianma/TDT4265Project/YOLO/naplab_data.yaml'
    # Project
    project = 'cityscapes_project'
    experiment = 'cityscapes'


    model.train(data = cityscapes,
                pretrained=False,
                epochs = 10,
                batch = 16,
                device = 0,
                imgsz = 640,
                patience = 5, 
                verbose = True,
                val = False,
                time = 0.1,
                )

    
    model.train(data = naplab,
                epochs = 10,
                batch = 16,
                device = 0,
                imgsz = 640,
                patience = 5, 
                verbose = True,
                val = True,
                time = 0.1
                )
