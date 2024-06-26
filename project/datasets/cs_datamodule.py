import lightning.pytorch as pl
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils, tv_tensors
from torchvision.transforms import v2, InterpolationMode
import torch
from torchvision.tv_tensors import BoundingBoxes


from datasets.cityscapes import Cityscapes


class CityscapesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=30,
                 num_workers=11,
                 data_root="/work/baardrw/cityscapesDataset",
                 label2idx=None,
                 valid_labels=None,
                 mode='fine',
                 device='cuda',
                 image_dimensions = [512, 1024],
                 resize_dims = [512, 512]
        ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.label2idx = label2idx
        self.mode = mode
        self.image_dimensions = image_dimensions
        self.resize_dims = resize_dims

        if valid_labels == None:
            self.valid_labels = [24, 25, 26, 27, 28, 33]
        else:
            self.valid_labels = valid_labels

    def prepare_data(self):
        # Download the dataset if needed (only using rank 1)
        pass

    def setup(self, stage=None):
        # Split the dataset into train and validation sets
        self.train_dataset = Cityscapes(
            root=self.data_root,
            mode=self.mode,
            target_type='polygon',
            split='train',
            transforms=self.get_transforms("train"),
            valid_labels=self.valid_labels,
            label2idx=self.label2idx
        )

        self.val_dataset = Cityscapes(
            root=self.data_root,
            mode=self.mode,
            target_type='polygon',
            split='val',
            transforms=self.get_transforms("val"),
            valid_labels=self.valid_labels,
            label2idx=self.label2idx
        )

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True,  collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        test_dataset = Cityscapes(
            root=self.data_root,
            mode=self.mode,
            target_type='polygon',
            split='test',
            transforms=self.get_transforms("test"),
            valid_labels=self.valid_labels,
            label2idx=self.label2idx,
        )
        
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=self.collate_fn)

    def target_transform(self, target):
        # print(target)
        return target

    def get_transforms(self, split, label=False):
        mean = [0.3090844516698354]
        std = [0.17752945677448584]

        shared_transforms = [
            
            v2.ToImage(),
            
            v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomCrop(size=(self.image_dimensions[0], self.image_dimensions[1])),
            
            v2.Resize(size=(self.resize_dims[0], self.resize_dims[1]), antialias=True),
            v2.ClampBoundingBoxes(),
            v2.SanitizeBoundingBoxes(),
           
        ]

        if split == "train":

            return v2.Compose([
                *shared_transforms,
                v2.RandomApply([v2.RandomRotation(degrees=5),  v2.ColorJitter(brightness=0.5, contrast= (1,10)), v2.RandomResize(min_size=256, max_size=1024)], p=0.3),
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
                # ...
            ])


if __name__ == '__main__':
    import torchvision
    VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
    STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

    # Test the data loader
    loader = CityscapesDataModule(
        # data_root='/home/bard/Documents/cityscapes',
        valid_labels=VALID_LABELS,
        label2idx=STR2IDX,
        image_dimensions=[512, 1024],
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

