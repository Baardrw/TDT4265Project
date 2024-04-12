import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils, tv_tensors
from torchvision.transforms import v2
import torch
from torchvision.tv_tensors import BoundingBoxes


from cityscapes import Cityscapes


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=30, num_workers=11, data_root="/home/bard/Documents/cityscapes", train_split_ratio=0.8, valid_labels = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.train_split_ratio = train_split_ratio
        
        if valid_labels == None:
            self.valid_labels = [24, 25, 26, 27, 28, 33]

    def prepare_data(self):
        # Download the dataset if needed (only using rank 1)
        pass

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        self.train_dataset = Cityscapes(
            root=self.data_root, mode='fine', target_type='polygon', split='train', transforms= self.get_transforms("train"))
        
        # change line 191 in cityscapes.py to target = Image.open(self.targets[index][i]).convert("L")
        # self.train_dataset[0][1]
        # self.train_dataset[0][0]

        
        # print(self.train_dataset[0][1]['labels'])
        # print(self.train_dataset[0][1]['masks'])

        self.val_dataset = Cityscapes(
            root=self.data_root, mode='fine', target_type='instance', split='val')
        
        
    def collate_fn(self, batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True,  collate_fn=lambda batch: tuple(zip(*batch)))

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_dataset = datasets.wrap_dataset_for_transforms_v2(datasets.Cityscapes(
            root=self.data_root, mode='fine', target_type='instance', split='test', transform=self.get_transforms("test")))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def target_transform(self, target):
        # print(target)
        return target
    
    def get_transforms(self, split, label=False):
        mean = [0.4914]
        std = [0.2023]

        shared_transforms = [
            v2.ToImage(),
            # v2.Normalize(mean, std)
        ]

        if split == "train":
            
            return v2.Compose([
                    # v2.ToTensor(),
                    v2.RandomResizedCrop(size=(512, 512), antialias=True),
                    v2.Grayscale(num_output_channels=1),
                    v2.ToDtype(torch.float32, scale=True),
                    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])                  
            
            return v2.Compose([
                *shared_transforms,
                v2.RandomCrop(32, padding=4, padding_mode='reflect'),
                v2.RandomHorizontalFlip(),
                # ...
            ])

        elif split == "val":
            return v2.Compose([
                *shared_transforms,
                # ...
            ])
        elif split == "test":
            return v2.Compose([
                *shared_transforms,
                # ...
            ])


if __name__ == '__main__':
    import torchvision
    
    
    # Test the data loader
    loader = CityscapesDataModule()
    loader.setup()
    print(len(loader.train_dataset))
    
    train_loader:DataLoader = loader.train_dataloader()
    # Get one batch from dataloader
    batch = next(iter(train_loader))
    # print(batch[1][0])
    # print(batch[1][1])    
    
    for i in range(len(batch[0])):
        # # transform train_ds[0][0] to uint8
        all_bbs = batch[1][i]['boxes']
        
        image_tensor = v2.functional.to_dtype(batch[0][i], torch.float32)
        image_uint8 = (image_tensor * 255).type(torch.uint8)
        
        img = utils.draw_bounding_boxes(image_uint8, all_bbs, width=5)
        #img = utils.draw_segmentation_masks(batch[0][0], batch[1][0] )
        import matplotlib.pyplot as plt
        # print(img.numpy().transpose(1, 2, 0).shape)
        # showable_im = img.numpy().transpose(1, 2, 0)
        # plt.imshow(img.numpy().transpose(1, 2, 0))
        plt.imsave(f"ims/test{i}.png", img.numpy().transpose(1, 2, 0))