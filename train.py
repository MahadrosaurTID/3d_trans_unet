import importlib
import sys
import torch
from src.models.unet3d import unet3d
from torchvision import transforms

from src.dataset.train_val_split import train_val_split
from src.losses.ce_dice_loss import CrossEntropyDiceLoss3D

from src.losses import dice_loss, region_based_loss, new_losses

from src.models.io_model import load_model
from src.train.trainer import Trainer, TrainerArgs
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from src.config import BratsConfiguration
from src.dataset.augmentations import color_augmentations, spatial_augmentations

from src.dataset.utils import dataset, visualization as visualization
from src.models.vnet import vnet, asymm_vnet
from src.logging_conf import logger
from src.dataset.loaders.brats_dataset import BratsDataset


def num_params(net_params):
    n_params = sum([p.data.nelement() for p in net_params])
    logger.info(f"Number of params: {n_params}")


# PARAMS
logger.info("Processing Parameters...")

config = BratsConfiguration(sys.argv[1])
model_config = config.get_model_config()
dataset_config = config.get_dataset_config()
basic_config = config.get_basic_config()

patch_size = config.patch_size
tensorboard_logdir = basic_config.get("tensorboard_logs")
checkpoint_path = model_config.get("checkpoint")
batch_size = dataset_config.getint("batch_size")
n_patches = dataset_config.getint("n_patches")
n_classes = dataset_config.getint("classes")
loss = model_config.get("loss")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"Device: {device}")

# DATASET
logger.info("Creating Dataset...")

data, _ = dataset.read_brats(dataset_config.get("train_csv"), lgg_only=dataset_config.getboolean("lgg_only"))
data_train, data_val = train_val_split(data, val_size=0.2)
data_train = data_train * n_patches
data_val = data_val * n_patches

n_modalities = dataset_config.getint("n_modalities")  # like color channels
sampling_method = importlib.import_module(dataset_config.get("sampling_method"))

transform = transforms.Compose([color_augmentations.RandomIntensityShift(),
                                color_augmentations.RandomIntensityScale(),
                                spatial_augmentations.RandomMirrorFlip(p=0.5),
                                spatial_augmentations.RandomRotation90(p=0.5)])

compute_patch = basic_config.getboolean("compute_patches")
train_dataset = BratsDataset(data_train, sampling_method, patch_size, compute_patch=compute_patch, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

val_dataset = BratsDataset(data_val, sampling_method, patch_size, compute_patch=compute_patch, transform=transform)
val_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if basic_config.getboolean("plot"):
    data_batch, labels_batch = next(iter(train_loader))
    data_batch.reshape(data_batch.shape[0] * data_batch.shape[1], data_batch.shape[2], data_batch.shape[3],
                       data_batch.shape[4], data_batch.shape[5])
    labels_batch.reshape(labels_batch.shape[0] * labels_batch.shape[1], labels_batch.shape[2], labels_batch.shape[3],
                         labels_batch.shape[4])

    print(data_batch.shape)
    logger.info('Plotting images')
    visualization.plot_batch_slice(data_batch, labels_batch, slice=30, save=True)

# MODEL
logger.info("Initiating Model...")

config_network = model_config["network"]
if config_network == "vnet":

    network = vnet.VNet(elu=model_config.getboolean("use_elu"),
                        in_channels=n_modalities,
                        classes=n_classes,
                        init_features_maps=model_config.getint("init_features_maps"))

elif config_network == "vnet_asymm":
    network = asymm_vnet.VNet(non_linearity=model_config.get("non_linearity"), in_channels=n_modalities, classes=n_classes,
                              init_features_maps=model_config.getint("init_features_maps"), kernel_size=model_config.getint("kernel_size"),
                              padding=model_config.getint("padding"))

elif config_network == "3dunet_residual":

    network = unet3d.ResidualUNet3D(in_channels=n_modalities, out_channels=n_classes, final_sigmoid=False,
                                    f_maps=model_config.getint("init_features_maps"), layer_order="gcr",
                                    num_levels=5, num_groups=8, conv_padding=1)

elif config_network == "3dunet":

    network = unet3d.ResidualUNet3D(in_channels=n_modalities, out_channels=n_classes, final_sigmoid=False,
                                    f_maps=model_config.getint("init_features_maps"), layer_order="crg",
                                    num_levels=4, num_groups=4, conv_padding=1)
else:
    raise ValueError("Bad parameter for network {}".format(model_config.get("network")))

num_params(network.parameters())

# TRAIN
logger.info("Start Training")
network.to(device)

optim = model_config.get("optimizer")

if optim == "SGD":
    optimizer = torch.optim.SGD(network.parameters(), lr=model_config.getfloat("learning_rate"),
                                momentum=model_config.getfloat("momentum"), weight_decay=model_config.getfloat("weight_decay"))
elif optim == "ADAM":
    optimizer = torch.optim.Adam(network.parameters(), lr=model_config.getfloat("learning_rate"), weight_decay=model_config.getfloat("weight_decay"), amsgrad=False)

else:
    raise ValueError("Bad optimizer. Current options: [SGD, ADAM]")

best_loss = 1000
if basic_config.getboolean("resume"):
    logger.info("Loading model from checkpoint..")
    model, optimizer, start_epoch, best_loss = load_model(network, checkpoint_path, device, optimizer, True)
    logger.info(f"Loaded model with starting epoch {start_epoch}")
else:
    start_epoch = 0

writer = SummaryWriter(tensorboard_logdir)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=model_config.getfloat("lr_decay"),
                                           patience=model_config.getint("patience"))

if loss == "dice":
    criterion = dice_loss.DiceLoss(classes=n_classes, eval_regions=model_config.getboolean("eval_regions"),
                                   sigmoid_normalization=True)

elif loss == "combined":
    # 0. back, 1: ncr, 2: ed, 3: et
    ce_weigh = torch.tensor([0.1, 0.35, 0.2, 0.35])
    criterion = CrossEntropyDiceLoss3D(weight=ce_weigh, classes=n_classes,
                                       eval_regions=model_config.getboolean("eval_regions"), sigmoid_normalization=True)
elif loss == "both_dice":
    criterion = region_based_loss.RegionBasedDiceLoss3D(classes=n_classes, sigmoid_normalization=True)

elif loss == "gdl":
    criterion = new_losses.GeneralizedDiceLoss()

else:
    raise ValueError(f"Bad loss value {loss}. Expected ['dice', combined]")

args = TrainerArgs(model_config.getint("n_epochs"), device, model_config.get("model_path"), loss)
trainer = Trainer(args, network, optimizer, criterion, start_epoch, train_loader, val_loader, scheduler, writer)
trainer.start(best_loss=best_loss)

print("Finished!")
