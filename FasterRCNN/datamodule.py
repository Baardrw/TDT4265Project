import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
from torchvision.transforms import v2
import torch


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=11, data_root="/work/baardrw/cityscapesDataset", train_split_ratio=0.8, valid_labels = None):
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
        self.train_dataset = datasets.wrap_dataset_for_transforms_v2(datasets.Cityscapes(
            root=self.data_root, mode='fine', target_type='instance', split='train', transform=self.get_transforms("train"), target_transform=self.get_target_transforms("train")))
        
        print(type((self.train_dataset[0][0])))
        print(type((self.train_dataset[0][1]['masks'][0])))
        
        self.val_dataset = datasets.wrap_dataset_for_transforms_v2(datasets.Cityscapes(
            root=self.data_root, mode='fine', target_type='instance', split='val'))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        test_dataset = datasets.wrap_dataset_for_transforms_v2(datasets.Cityscapes(
            root=self.data_root, mode='fine', target_type='instance', split='test', transform=self.get_transforms("test")))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False)

    def get_target_transforms(self, split):
        # Convert polygon to bounding box
        pass
            
    
    def get_transforms(self, split):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        shared_transforms = [
            v2.ToImage(),
            # v2.Normalize(mean, std)
        ]

        if split == "train":
            return v2.Compose(
            [
                v2.to_tensor(),
                v2.RandomPhotometricDistort(p=1.0),
                v2.RandomRotation(degrees=10)
            ]
            )                   
            
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
    # Test the data loader
    loader = CityscapesDataModule()
    loader.setup()
    
    train_loader:DataLoader = loader.train_dataloader()
    
    
    for test_image, test_target in train_loader:
        print(test_image.shape, test_target['masks'].shape)
        break
   
    # # transform train_ds[0][0] to uint8
    # all_masks = train_ds[0][1]['masks'].squeeze(
    #     1)  # Remove channel dimension
    

    # image_tensor = v2.functional.to_dtype(train_ds[0][0], torch.float32)
    # image_uint8 = (image_tensor * 255).type(torch.uint8)
    
    # # img = utils.draw_segmentation_masks(image_uint8, all_masks)
    # img = utils.draw_segmentation_masks(image_uint8, all_masks)

    # import matplotlib.pyplot as plt
    # print(img.numpy().transpose(1, 2, 0).shape)
    # showable_im = img.numpy().transpose(1, 2, 0)

    # plt.imshow(showable_im)
    # plt.show()()
