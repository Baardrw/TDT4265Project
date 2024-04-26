import json
import math
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchvision

from PIL import Image

from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms import v2, functional 
import torchvision.tv_tensors


### === Modified code from torchvision.datasets.cityscapes, because the code did not behave as expected  === ###

class NapLab(VisionDataset):
    
    names = [
        "background",
        "truck",
        "bus",
        "bicycle",
        "scooter",
        "person",
        "rider",
        "car"
    ]
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        train_val_split: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        image_dimensions: Tuple[int, int] = (128, 1024),
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.split = split
        self.targets_dir = os.path.join(self.root, )
        self.split = split
        self.images = [os.path.join(self.root, line.strip()) for line in open(os.path.join(self.root, f"train.txt"))]
        self.targets = [image.replace('images', 'labels_yolo_v1.1').replace('PNG', 'txt') for image in self.images]
        self.label2idx = {cls: idx for idx, cls in enumerate(self.names)}
        self.image_height, self.image_width = image_dimensions
        
        num_train = math.floor(len(self.images) * train_val_split)
        if split == 'train':
            self.images = self.images[:num_train]
            self.targets = self.targets[:num_train]
        elif split == 'all':
            pass
        else:
            self.images = self.images[num_train:]
            self.targets = self.targets[num_train:]
                
        
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of labels and bounding boxes
        """

        org_image = Image.open(self.images[index]).convert("L")
        image = torchvision.tv_tensors.Image(org_image)
        image = functional.equalize(image)
        
        labels = []
        bounding_boxes = []
        
        labels.append(0) # Background label
        bounding_boxes.append((0, 0, self.image_height, self.image_width)) # Probs doesnt matter if it is much bigger than the window anyway
        
        target = self.__load_target(self.targets[index])
        
        for cls, x1, y1, x2, y2 in target:
            
            assert x1 <= x2
            assert y1 <= y2
            
            labels.append(int(cls))
            bounding_boxes.append((x1, y1, x2, y2))
                        
        
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'labels': labels,
            'boxes': BoundingBoxes(bounding_boxes, format='XYXY', canvas_size=[self.image_height, self.image_width])
        }
   
        
        if self.transforms is not None:
            # target is a list of tuple[labels, bounding_boxes]
            image, target = self.transforms(image, target)
            
            # Remove the background label and bounding box
        # target['labels'] = target['labels'][0:]
        # target['boxes'] = target['boxes'][0:]
        if self.split == 'test':
            return image
        
        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def __load_target(self, path: str):
        """
        Loads bbs from file, each line corresponds to one object. The object is represented by its id and yolo v1.1 value.
        eg. 0 0.442876 0.735078 0.037900 0.230156
        """
        
        for line in open(path):
            line = line.strip().split()
            cls = int(line[0])
            if cls == 0:
                cls = 7
            
        
            x, y, w, h = map(float, line[1:])
            
            x*= self.image_width
            y*= self.image_height
            w*= self.image_width
            h*= self.image_height
            
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            
            yield cls, x1, y1, x2, y2
            


if __name__ == '__main__':
    import torchvision

    dataset = NapLab(root='/datasets/tdt4265/ad/NAPLab-LiDAR', split='all')

    mean = 0
    std = 0
    
    for image in dataset:
        numpy_image = image[0].numpy().transpose(1, 2, 0)
        mean += np.mean(numpy_image.flatten())
        std += np.std(numpy_image.flatten())
    
    mean /= len(dataset)
    mean /= 255
    
    std /= len(dataset)
    std /= 255
    
    print(mean, std)