# import required functions, classes
import munch
import sahi
# from sahi.predict import get_sliced_prediction, predict, get_prediction
# from sahi.utils.file import download_from_url
# from sahi.utils.cv import read_image
from IPython.display import Image
# import sahi.predict
import sahi.predict as pred
import yaml
from datasets.nl_datamodule import NapLabDataModule
from trainer import LitModel

config = munch.munchify(yaml.load(open("config.yaml"), Loader=yaml.FullLoader))

if __name__ == '__main__':
    model = LitModel.load_from_checkpoint(checkpoint_path=config.checkpoint_path, config=config)
    dm = NapLabDataModule(
           batch_size=1,
           num_workers=1,
           data_root=config.data_root,
           image_dimensions=[128, 1024],
           resize_dims=[128,1024]
       )
    
    test_image = dm.test_dataloader().dataset[0]
    from torchvision.transforms import ToPILImage
    test_image = ToPILImage()(test_image)

    result = pred.get_sliced_prediction(
        image= test_image,
        slice_height=128,
        slice_width=256,
        detection_model= model,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.2,
    )
    
    result.export_visuals('demo/')
    
