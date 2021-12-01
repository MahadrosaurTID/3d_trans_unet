import torch
import torch.nn as nn

from src.models.unet3d.building_blocks import Encoder, Decoder, DoubleConv, ExtResNetBlock
from src.models.Transformer import TransformerModel
from torchsummary import summary
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


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Abstract3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info

        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)

        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False

        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):

        super(Abstract3DUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):

            if i == 0:
                encoder = Encoder(in_channels, out_feature_num,
                                  apply_pooling=False,  # skip pooling in the first encoder
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  padding=conv_padding)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num,
                                  basic_module=basic_module,
                                  conv_layer_order=layer_order,
                                  conv_kernel_size=conv_kernel_size,
                                  num_groups=num_groups,
                                  pool_kernel_size=pool_kernel_size,
                                  padding=conv_padding)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        # -------------------------- Haris ---------------------------------------------
        self.transformer = TransformerModel(dim=4, heads=2, depth=128, mlp_dim=8)
        # ------------------------------------------------------------------------------

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            if basic_module == DoubleConv:
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            else:
                in_feature_num = reversed_f_maps[i]

            out_feature_num = reversed_f_maps[i + 1]

            decoder = Decoder(in_feature_num, out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)

            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # ---------------- Haris -------------------
        last_map = encoders_features[0]
        last_map, x_int = self.transformer(last_map)
        encoders_features[0] = last_map
        # ------------------------------------------

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        scores = self.final_activation(x)

        return x, scores


class UNet3D(Abstract3DUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.
    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, conv_padding=1, **kwargs):

        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels, final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv, f_maps=f_maps, layer_order=layer_order,
                                     num_groups=num_groups, num_levels=num_levels,
                                     conv_padding=conv_padding, **kwargs)

    def test(self):
        classes = 4
        in_channels = 4
        input_tensor = torch.rand(1, in_channels, 32, 32, 32)
        ideal_out = torch.rand(1, classes, 32, 32, 32)

        out_pred, out_scores = self.forward(input_tensor)
        assert ideal_out.shape == out_pred.shape

        print("UNet3D test is complete")


class ResidualUNet3D(Abstract3DUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ExtResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock, f_maps=f_maps, layer_order=layer_order,
                                             num_groups=num_groups, num_levels=num_levels, conv_padding=conv_padding, **kwargs)
        print()

    def test(self):
        classes = 4
        in_channels = 4
        input_tensor = torch.rand(1, in_channels, 155, 240, 240)
        ideal_out = torch.rand(1, classes, 32, 32, 32)

        # MAHAD
        config = BratsConfiguration('/home/mahad/PycharmProjects/mri-braintumor-segmentation/resources/config.ini')
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

        for dat, labels in train_loader:
            inputs = dat.float()
            targets = labels.float()
            print(f"Input shape : {inputs.shape}")
            inputs.require_grad = True
            predictions, _ = self.forward(inputs)
            print(f"prediction shape : {predictions.shape}")
        out_pred, _ = self.forward(input_tensor)
        assert ideal_out.shape == out_pred.shape

        print("ResidualUNet3D test is complete")


if __name__ == "__main__":
    import numpy as np
    net = ResidualUNet3D(in_channels=4, out_channels=4, f_maps=16)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)
    net.test()

    # unet = UNet3D(in_channels=4, out_channels=4, f_maps=16, final_sigmoid=True, layer_order='crg',
    #               num_groups=8, num_levels=4, conv_padding=1)
    # model_parameters = filter(lambda p: p.requires_grad, unet.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(params)
    # unet.test()
