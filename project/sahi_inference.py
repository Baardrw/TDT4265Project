# import required functions, classes
import math
from typing import List
import munch
import sahi
# from sahi.predict import get_sliced_prediction, predict, get_prediction
# from sahi.utils.file import download_from_url
# from sahi.utils.cv import read_image
from sahi import AutoDetectionModel
import sahi.postprocess
import sahi.postprocess.utils
import sahi.prediction
import sahi.utils
import sahi.utils.coco
import torch
from PIL import Image
# import sahi.predict
import sahi.predict as pred
import yaml
from datasets.nl_datamodule import NapLabDataModule
from trainer import LitModel
from sahi.models import torchvision
import torchmetrics.detection.mean_ap


config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))



def nms(object_predictions: List[sahi.prediction.ObjectPrediction], thresh: float) -> List[sahi.prediction.ObjectPrediction]:
    """Performs NMS on the given object predictions. and returns the pruned list of predictions."""
    
    def collapse_bus_truck_and_car(d1, d2):
        c1 = d1.category.id
        c2 = d2.category.id
        
        if c1 == 1 or c1 == 2 or c1 == 7:
            c1 = 1
        
        if c2 == 1 or c2 == 2 or c2 == 7:
            c2 = 1
        
        if c1 == c2: # Best bet is to default to car if the classes are the same
            d1.category.id = 7
            d2.category.id = 7
            
            return True

        return False
        
    
    # Sort the predictions by confidence
    object_predictions.sort(key=lambda x: x.score.value, reverse=True)
    
    detections = []
    
    while len(object_predictions) > 0:
        # Pick the prediction with the highest confidence
        best_detection = object_predictions.pop(0)
        
        # Remove all the predictions that have an IoU greater than the threshold and same class
        # Also treats buss car and truck as the same class because the model is not able to distinguish between them well
        object_predictions = [x for x in object_predictions if sahi.postprocess.utils.calculate_bbox_iou(best_detection, x) < thresh or (x.category.id != best_detection.category.id and not collapse_bus_truck_and_car(best_detection, x))]
        
        # # We 
        
        
        detections.append(best_detection)
    
    
    return detections
    
    
def aspect_ratio_filtering(object_predictions: List[sahi.prediction.ObjectPrediction]) -> List[sahi.prediction.ObjectPrediction]:
    """Performs aspect ratio filtering on the given object predictions. and returns the pruned list of predictions."""
    
    MIN_ASPECT_RATIO_BY_CLASS = {
        0: 0, # Background
        1: 0.442393055489467, # Truck
        2: 0.43343741806307057, # Bus
        3: 0.5486895366027612, # Scooter
        4: 0.22806264750053634, # Bicycle
        5: 0.09697331470201191, # Person
        6: 0.08970891740724066, # Rider
        7: 0.11368660628503735 # Car
    }
    
    MAX_ASPECT_RATIO_BY_CLASS = {
        0: math.inf, # Background
        1: 2.7954859530303824, # Truck
        2: 3.115624, # Bus
        3: 0.5486895366027612, # Scooter
        4: 2.5844772623772037, # Bicycle
        5: 0.7107872611464968, # Person
        6: 0.8600745875560987, # Rider
        7: 11.426960636262963 # Car
    }
    
    detections = []
    
    for detection in object_predictions:
        # Calculate the aspect ratio of the bounding box
        width = detection.bbox.maxx - detection.bbox.minx
        height = detection.bbox.maxy - detection.bbox.miny
        aspect_ratio = width / height
        
        if aspect_ratio >= MIN_ASPECT_RATIO_BY_CLASS[detection.category.id] and aspect_ratio <= MAX_ASPECT_RATIO_BY_CLASS[detection.category.id]:
            detections.append(detection)
    
    print(f"Aspect ratio filtered: {len(object_predictions) - len(detections)} detections.")
    
    return detections
    
    
def area_filtering(object_predictions: List[sahi.prediction.ObjectPrediction]) -> List[sahi.prediction.ObjectPrediction]:
    MIN_AREA_BY_CLASS = {
        0: 0, # Background
        1: 156.85801934847998, # Truck
        2: 21.733349130239997, # Bus
        3: 54.56016048128, # Scooter
        4: 21.530411532288, # Bicycle
        5: 17.866427990015996, # Person
        6: 20.357349113855996, # Rider
        7: 9.990979190784001, # Car
    }
    
    MAX_AREA_BY_CLASS = {
        0: math.inf, # Background
        1: 45145.16963006874, # Truck
        2: 51046.383616, # Bus
        3: 801.725240967168, # Scooter
        4: 933.5661298974721, # Bicycle
        5: 2964.890429423616, # Person
        6: 4190.324063993856, # Rider
        7: 32424.301076152322, # Car
    }
    
    detections = []
    
    for detection in object_predictions:
        width = detection.bbox.maxx - detection.bbox.minx
        height = detection.bbox.maxy - detection.bbox.miny
        area = width * height
        
        if area >= MIN_AREA_BY_CLASS[detection.category.id] and area <= MAX_AREA_BY_CLASS[detection.category.id]:
            detections.append(detection)
    
    print(f"Area filtered: {len(object_predictions) - len(detections)} detections.")
    return detections


def sahi_test():
    model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    
    val_map = torchmetrics.detection.mean_ap.MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox", 
        )
    
    dm = NapLabDataModule( # No transformations
           batch_size=1,
           num_workers=1,
           data_root=config.data_root,
           image_dimensions=[128, 1024],
           resize_dims=[128,1024]
        )
    
    dm.setup()
    
    val_loader = dm.val_dataloader() # Used to get the ground truths
    image_paths = dm.test_dataloader().dataset.images
    inference_speed = []
    
    # This is a stupid way of doing it but its the only way to make SAHI work while getting the labels
    for i, batch in enumerate(val_loader):
        target = batch[1]
        image = image_paths[i]
        
        prediction: sahi.prediction.PredictionResult = sahi_inference([image], model, image_index=i)
        prediction_list: List[sahi.prediction.ObjectPrediction] = prediction.object_prediction_list
        
        inference_speed.append(prediction.durations_in_seconds['prediction'])
        
        pred_dict = {
            'boxes': [],
            'labels': [],
            'scores': []
        }
        
        for pred in prediction_list:
            
            pred_dict['boxes'].append([pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy])
            pred_dict['labels'].append(pred.category.id)
            pred_dict['scores'].append(pred.score.value)
            
        # boxes should be FloatTensor[N, 4]
        pred_dict['boxes'] = torch.tensor(pred_dict['boxes'])
        pred_dict['labels'] = torch.tensor(pred_dict['labels'])
        pred_dict['scores'] = torch.tensor(pred_dict['scores'])
        
        val_map.update([pred_dict], target)
    
    val_dict = val_map.compute()
    print(val_dict)
    print(f"Average inference speed: {sum(inference_speed) / len(inference_speed)}")
        
    
    


def sahi_inference(images, model, image_index=0):
    for i, path in enumerate(images):
    
        result: sahi.prediction.PredictionResult = pred.get_sliced_prediction(
            image= path,
            slice_height=128,
            slice_width=256,
            detection_model= model,
            overlap_height_ratio=0.0,
            overlap_width_ratio=0.5,
            verbose=True,
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.6, # 0.1285
            )
        
        # Prune the predictions on confidence
        result.object_prediction_list = list(filter(lambda x: x.score.value > 0.5, result.object_prediction_list))
        
        # Aspect ratio filtering
        result.object_prediction_list = aspect_ratio_filtering(result.object_prediction_list)
        
        # Area filtering
        # result.object_prediction_list = area_filtering(result.object_prediction_list)
        
        # Flying car filter
        
        
        # Non max suppresion
        result.object_prediction_list = nms(result.object_prediction_list, 0.4)
        

        result.export_visuals('demo/', file_name=f'{image_index}.png', rect_th=1, text_size=0.3)
        
        model.reset_sahi_detections()
        print(image_index)
        return result
    
if __name__ == '__main__':
    
    sahi_test()
    exit()
    
    
    model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    dm = NapLabDataModule(
           batch_size=1,
           num_workers=1,
           data_root=config.data_root,
           image_dimensions=[128, 1024],
           resize_dims=[128,1024]
        )
    
    dm.setup()
    test_data = dm.test_dataloader().dataset
    
    sahi_inference(test_data.images)
    
  
# Faster RCNN  
# {'map': tensor(0.1285), 'map_50': tensor(0.2728), 'map_75': tensor(0.0960), 'map_small': tensor(0.1383), 'map_medium': tensor(0.2864), 'map_large': tensor(0.1496), 'mar_1': tensor(0.0758), 'mar_10': tensor(0.1553), 'mar_100': tensor(0.1826), 'mar_small': tensor(0.2071), 'mar_medium': tensor(0.3719), 'mar_large': tensor(0.1789), 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.), 'classes': tensor([0, 2, 3, 4, 5, 6, 7], dtype=torch.int32)}
# Average inference speed: 0.13530259495332173