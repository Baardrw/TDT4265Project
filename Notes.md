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


Maybe add more transforms to fit better to naplab dataset


## Fine tuning

128x512 is good size
regular cross entropy loss is best at least in the start, not sure about later on, no progressive resizing seems promising, although prog res might be good 