import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader
from torchvision import utils
from torchvision.transforms import v2
import torch


try:
    from datasets.naplab import NapLab # to allow running the script from the project root
except ImportError:
    from naplab import NapLab


class NapLabDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=30,
                 num_workers=11,
                 data_root="/datasets/tdt4265/ad/NAPLab-LiDAR",
                 image_dimensions = [128, 256],
                 resize_dims = [512 , 512]
                ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.image_dimensions = image_dimensions
        self.resize_dims = resize_dims


    def prepare_data(self):
        # Download the dataset if needed (only using rank 1)
        pass

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        self.train_dataset = NapLab(
            root=self.data_root,
            split='train',
            transforms=self.get_transforms("train"),
        )

        self.val_dataset = NapLab(
            root=self.data_root,
            split='val',
            transforms=self.get_transforms("val"),
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True,  collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        test_dataset = NapLab(
            root=self.data_root,
            split='test',
            transforms=self.get_transforms("test"),
        )
        
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=self.collate_fn)


    def get_transforms(self, split, label=False):
        mean = [0.5085652989351241] # taken from data analysis of naplab dataset
        std = [0.2970477123406435]

        shared_transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(size=(self.image_dimensions[0], self.image_dimensions[1])),
            v2.Resize(size=(self.resize_dims[0], self.resize_dims[1]), antialias=True),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
        ]

        if split == "train":

            return v2.Compose([
                *shared_transforms,
                v2.RandomApply([v2.RandomRotation(degrees=15), v2.RandomHorizontalFlip(),v2.RandomPhotometricDistort(), v2.RandomVerticalFlip(p=0.2), v2.RandomHorizontalFlip(p=1.0)], p=0.5),
                v2.SanitizeBoundingBoxes(),
                v2.Normalize(mean=mean, std=std),
                
                
            ])

        elif split == "val":
            return v2.Compose([
                *shared_transforms,
                v2.Normalize(mean=mean, std=std),
                
                # ...
            ])
        elif split == "test":
            return v2.Compose([
                *shared_transforms,
                v2.Normalize(mean=mean, std=std),
                
            ])


if __name__ == '__main__':
    import torchvision

    # Test the data loader
    loader = NapLabDataModule(
        # data_root='/home/bard/Documents/cityscapes',
        batch_size=30,
        num_workers=0,
    )
    loader.setup()
    print(len(loader.train_dataset))

    train_loader: DataLoader = loader.train_dataloader()
    # Get one batch from dataloader
    batch = next(iter(train_loader))
    # print(batch[1][0])
    # print(batch[1][1])
    
    mean = 0
    std = 0

    for i in range(len(batch[0])):
        # # transform train_ds[0][0] to uint8
        all_bbs = batch[1][i]['boxes']
        print(len(all_bbs))
        labels = batch[1][i]['labels']
        
     
            

        image_tensor = v2.functional.to_dtype(batch[0][i], torch.float32)

        mean += np.mean(image_tensor.numpy().transpose(1, 2, 0).flatten())
        std += np.std(image_tensor.numpy().transpose(1, 2, 0).flatten())
        
        image_uint8 = (image_tensor * 255).type(torch.uint8)

        img = utils.draw_bounding_boxes(image_uint8, all_bbs, width=1)
        # img = utils.draw_segmentation_masks(batch[0][0], batch[1][0] )
        import matplotlib.pyplot as plt
        print(img.numpy().transpose(1, 2, 0).shape)
        print(max(img.numpy().transpose(1, 2, 0).flatten()))
        
        # showable_im = img.numpy().transpose(1, 2, 0)
        # plt.imshow(img.numpy().transpose(1, 2, 0))
        if img.numpy().transpose(1, 2, 0).shape[2] == 1:
            plt.imsave(f"ims/train{i}.png", img.numpy().transpose(1, 2, 0)[:, :, 0], cmap='gray')
        else:
            plt.imsave(f"ims/train{i}.png", img.numpy().transpose(1, 2, 0))
         
    mean /= len(batch[0])
    std /= len(batch[0])
       
    print("Mean: ", mean)
    print("Std: ", std)

