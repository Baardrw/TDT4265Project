import json
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
from torchvision.transforms import v2
import torchvision.tv_tensors


### === Modified code from torchvision.datasets.cityscapes, because the code did not behave as expected  === ###

class Cityscapes(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        valid_labels: Optional[List[str]] = ['car', 'person'],
        label2idx: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        self.valid_labels = valid_labels
        
        if label2idx is not None:
            self.label2idx = label2idx
        else:
            self.label2idx = {cls: idx for idx, cls in enumerate(self.valid_labels)}

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = "Unknown value '{}' for argument split if mode is '{}'. Valid values are {{{}}}."
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [
            verify_str_arg(value, "target_type", ("instance", "semantic", "polygon", "color"))
            for value in self.target_type
        ]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == "train_extra":
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainextra.zip")
            else:
                image_dir_zip = os.path.join(self.root, "leftImg8bit_trainvaltest.zip")

            if self.mode == "gtFine":
                target_dir_zip = os.path.join(self.root, f"{self.mode}_trainvaltest.zip")
            elif self.mode == "gtCoarse":
                target_dir_zip = os.path.join(self.root, f"{self.mode}.zip")

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
                    )
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)
                
    def create_bb_dataset(self):
        if self.target_type != ['polygon']:
            raise ValueError(f"Only target_type='polygon' is supported for this method, type is {self.target_type}")
        
        
        for target in self.targets:
            target = target[0]
            
            t = self._load_json(target)
            t = self._filter_classes(t, self.valid_labels)
            
            labels = []
            bounding_boxes = []
            
            labels.append('background')
            bounding_boxes.append((0, 0, 2048, 1024))
            
            for object in t['objects']:
                labels.append(object['label'])
                polygon = object['polygon']
                
                min_x = min(polygon, key=lambda x: x[0])[0]
                min_y = min(polygon, key=lambda x: x[1])[1]
                max_x = max(polygon, key=lambda x: x[0])[0]
                max_y = max(polygon, key=lambda x: x[1])[1]
                
                if min_x == max_x or min_y == max_y:
                    print(f"Invalid bounding box: {min_x}, {min_y}, {max_x}, {max_y}")
                    continue
                
                bounding_boxes.append((min_x, min_y, max_x, max_y))
            
            # Transform string labels to integer labels
            labels = [self.label2idx[label] for label in labels]
            
            # Make the bounding box and id into a json dict and save it to the same directory as the polygon
            target_dict = {
                    'labels': labels,
                    'boxes': bounding_boxes
                }
            
            json_str = json.dumps(target_dict)
            target_name = "{}_{}".format(
                        target.split(self._get_target_suffix(self.mode, 'polygon'))[0], self._get_target_suffix(self.mode, 'boxes')
                    )
            
            # Save the json dict to a new file 
            with open(target_name, 'w') as file:
                file.write(json_str)
            

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """

        org_image = Image.open(self.images[index]).convert("RGB")
        image = torchvision.tv_tensors.Image(org_image)
        
        targets: Any = []
        labels = []
        
        
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
                target = self._filter_classes(target, self.valid_labels)
                
                labels = []
                bounding_boxes = []
                
                labels.append('background')
                bounding_boxes.append((0, 0, 2048, 1024))
                
                for object in target['objects']:
                    labels.append(object['label'])
                    polygon = object['polygon']
                    
                    min_x = min(polygon, key=lambda x: x[0])[0]
                    min_y = min(polygon, key=lambda x: x[1])[1]
                    max_x = max(polygon, key=lambda x: x[0])[0]
                    max_y = max(polygon, key=lambda x: x[1])[1]
                    
                    if min_x == max_x or min_y == max_y:
                        print(f"Invalid bounding box: {min_x}, {min_y}, {max_x}, {max_y}")
                        continue
                    
                    bounding_boxes.append((min_x, min_y, max_x, max_y))
                
                # Transform string labels to integer labels
                labels = [self.label2idx[label] for label in labels]
                
                # Transform labels to int64 tensor
                labels = torch.tensor(labels, dtype=torch.int64)
                
                target = {
                    'labels': labels,
                    'boxes': BoundingBoxes(bounding_boxes, format='XYXY', canvas_size=[1024, 2048])
                }
                
                # if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
                #     target['boxes'] = torch.zeros((0,4),dtype=torch.float32)
            
            else:
                target = Image.open(self.targets[index][i]).convert("L")
                # target_uint8 = np.array(target, dtype=np.uint8)
                # target = Image.fromarray(target_uint8)

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        
        
        if self.transforms is not None:
            # target is a list of tuple[labels, bounding_boxes]
            image, target = self.transforms(image, target)
            
            # Remove the background label and bounding box
            target['labels'] = target['labels'][1:]
            target['boxes'] = target['boxes'][1:]
        
        if self.split == 'test':
            return image
            
        return image, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return "\n".join(lines).format(**self.__dict__)

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data
    
    
    def _filter_classes(self, data, valid_labels):
        # Parse the JSON string into a Python dictionary
        
        # Filter out objects with unwanted classes
        data['objects'] = [obj for obj in data['objects'] if obj['label'] in valid_labels]
        
        return data


    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        elif target_type == "boxes":
            return f"{mode}_boxes.json"
        else:
            return f"{mode}_polygons.json"



if __name__ == '__main__':
    import torchvision
    VALID_LABELS = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'person', 'rider', 'background']
    STR2IDX = {cls: idx for idx, cls in enumerate(VALID_LABELS)}

    dataset = Cityscapes('/work/baardrw/cityscapesDataset', split='train', mode='fine',
                         target_type='polygon', valid_labels=VALID_LABELS, label2idx=STR2IDX)
    
    dataset.create_bb_dataset()