import torch
import segmentation_models_pytorch as smp

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ["nodamage", "minor-damage", "major-damage", "destroyed"]
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
BATCH_SIZE = 16

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER,
    in_channels=3,
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES)+1, 
    activation=ACTIVATION,
)

# Set flag to train the model or not. If set to 'False', only prediction is performed (using an older model checkpoint)
TRAINING = True

# Set num of epochs
EPOCHS = 15

# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define loss function
loss = smp.losses.JaccardLoss(mode="multiclass")
loss.__name__ = 'jaccard_loss'

# define metrics
metrics = [
    smp.utils.metrics.IoU(),
    smp.utils.metrics.Fscore()
]

# define optimizer
optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])

# define learning rate scheduler (not used in this NB)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)
