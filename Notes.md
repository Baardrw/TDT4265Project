<<<<<<< HEAD
## Adapting pretrained model to Grayscale images

2 approaches:
1) Adding additional channels to each greyscale image
2) Modifying the first convolutional layer of the pretrained network

https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a

=======
## Training of Mask R-CNN on City scapes

Implementation: We apply our Mask R-CNN models with
the ResNet-FPN-50 backbone; we found the 101-layer
counterpart performs similarly due to the small dataset size.
We train with image scale (shorter side) randomly sampled
from [800, 1024], which reduces overfitting; inference is on
a single scale of 1024 pixels. We use a mini-batch size of
1 image per GPU (so 8 on 8 GPUs) and train the model
for 24k iterations, starting from a learning rate of 0.01 and
reducing it to 0.001 at 18k iterations. It takes ∼4 hours of
training on a single 8-GPU machine under this setting.
>>>>>>> 4998a77631ef37a54e31c31b7b625eaf3a698ddf
