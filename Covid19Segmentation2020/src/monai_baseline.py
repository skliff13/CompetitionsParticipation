# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import types
import glob
import logging
import os
import shutil
import sys

from dotenv import load_dotenv
from ignite.engine.events import Events
import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
import wandb
from ignite.contrib.handlers.wandb_logger import *

import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadNiftid,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    Compose
)
from monai.data import CacheDataset, PersistentDataset, DataLoader, Dataset, NiftiSaver
from monai.losses import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from torchsummary import summary

from vgg16j import VGG16J, VGG19J


def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        # Spacingd(keys, pixdim=(1, 1, 4.0), mode=(
        Spacingd(keys, pixdim=(1.25, 1.25, 5), mode=(
            "bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0,
                             a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if args.imn_norm == '1':
        xforms.append(NormalizeIntensityd(keys[0], subtrahend=0.449, divisor=0.226))
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1),
                            mode="reflect"),  # ensure at least 192, 192
                RandAffined(
                    keys,
                    prob=0.30,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode='zeros',
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(
                    192, 192, 32), num_samples=4),
                RandAdjustContrastd(keys[0], prob=0.3, gamma=(0.8, 1.2)),
                RandGaussianSmoothd(keys[0], sigma_z=(0, 0)),
                RandGaussianSharpend(keys[0], sigma1_z=(0, 0), sigma2_z=(0, 0)),
                RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return Compose(xforms)


def get_net():
    """returns a unet model instance."""

    if args.model == 'unet':
        raise NotImplementedError("UNet is not implemented in this version of the script")
    elif args.model == 'vgg16j':
        net = VGG16J(cut_layers=1, down_conv_z=False, up_conv_z2=True)
    elif args.model == 'vgg19j':
        net = VGG19J(cut_layers=1, down_conv_z=False, up_conv_z2=True)
    else:
        raise ValueError(f'Unknown network model {args.model}')
    return net


def get_inferer(_mode=None, sw_batch_size=1):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 32)
    overlap = 0.5
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(
            y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + 10 * cross_entropy


class CELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(
            y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy


def load_split(images, labels, split_version):
    def path_to_id(path):
        return os.path.split(path)[-1].replace('_ct.nii.gz', '').replace('_seg.nii.gz', '')

    train_ids = set(np.loadtxt(os.path.join('metadata', f'Train_ids_v{split_version}.txt'), dtype=str))
    val_ids = set(np.loadtxt(os.path.join('metadata', f'Val_ids_v{split_version}.txt'), dtype=str))

    train_images = [p for p in images if path_to_id(p) in train_ids]
    train_labels = [p for p in labels if path_to_id(p) in train_ids]
    val_images = [p for p in images if path_to_id(p) in val_ids]
    val_labels = [p for p in labels if path_to_id(p) in val_ids]

    return train_images, train_labels, val_images, val_labels


def train(data_folder=".", model_folder="runs", cache_dir="cache", gpu="cuda"):
    """run a training pipeline."""
    hyperparameter_defaults = dict(
        batch_size=1,
        val_batch_size=1,
        epochs=500,
        lr=args.lr,
        log_interval=1,
        spatial_size=(192, 192, 32),
        pixdim=(1.25, 1.25, 5),
        features=(32, 64, 128, 256, 512, 32)
    )

    images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii*")))
    labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii*")))

    if args.demo != '0':
        images, labels = images[:10], labels[:10]

    logging.info(
        f"training: image/label ({len(images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    if args.split_version == '0':
        train_frac, val_frac = 0.8, 0.2
        n_train = int(train_frac * len(images)) + 1
        n_val = min(len(images) - n_train, int(val_frac * len(images)))
        logging.info(
            f"training: train {n_train} val {n_val}, folder: {data_folder}")
        train_images, train_labels = images[:n_train], labels[:n_train]
        val_images, val_labels = images[-n_val:], labels[-n_val:]
    else:
        train_images, train_labels, val_images, val_labels = load_split(images, labels, args.split_version)

    train_files = [{keys[0]: img, keys[1]: seg}
                   for img, seg in zip(train_images, train_labels)]
    val_files = [{keys[0]: img, keys[1]: seg}
                 for img, seg in zip(val_images, val_labels)]

    # create a training data loader
    batch_size = hyperparameter_defaults['batch_size']
    num_workers = 8
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = get_net().to(device)

    summary(net, input_size=(1, 192, 192, 32))

    max_epochs, lr = 500, hyperparameter_defaults['lr']
    logging.info(f"epochs {max_epochs}, lr {lr}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training
    val_post_transform = monai.transforms.Compose([AsDiscreted(
        keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)])
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={
                        "net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(sw_batch_size=batch_size),
        post_transform=val_post_transform,
        key_val_metric={"val_mean_dice": MeanDice(
            include_background=False, output_transform=lambda x: (x["pred"], x["label"]))},
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss",
                     output_transform=lambda x: x["loss"]),
    ]
    losses = {'dice_ce': DiceCELoss(), 'ce': CELoss(), 'dice': DiceLoss(to_onehot_y=True, softmax=True)}
    loss_function = losses[args.loss]
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss_function,
        inferer=get_inferer(sw_batch_size=batch_size),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    short_trainer = SupervisedTrainer(
        device=device,
        max_epochs=5,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=loss_function,
        inferer=get_inferer(sw_batch_size=batch_size),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )

    # WANDB logger
    # Get metrics in Weights & Biases
    wandb_logger = WandBLogger(
        project="covid-segmentation",
        name=os.path.split(args.model_folder)[-1],
        config=hyperparameter_defaults,
        tags=["pytorch-ignite", "covid"]
    )

    wandb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"loss": loss}
    )

    wandb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="training",
        metric_names=["val_mean_dice"],
        global_step_transform=lambda *_: trainer.state.iteration,
    )

    wandb_logger.attach_opt_params_handler(
        trainer,
        event_name=Events.ITERATION_STARTED,
        optimizer=opt,
        param_name='lr'  # optional
    )

    wandb_logger.watch(net)

    if args.start_weights:
        logging.info(f"using {args.start_weights}.")
        net.load_state_dict(torch.load(args.start_weights, map_location=device))

    if net.__class__ in {VGG16J, VGG19J} and not args.start_weights:
        print('Freezing pretrained layers')
        net.freeze_pretrained()
        short_trainer.run()
        print('Unfreezing pretrained layers')
        net.unfreeze_pretrained()
        trainer.run()
    else:
        trainer.run()


def infer(data_folder=".", model_folder="runs", prediction_folder="output", gpu="cuda"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=20,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(
                f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(
                    keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(
                        _img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

    def copy_files(prefix, suffix, submission_dir):
        files = glob.glob(os.path.join(prediction_folder, f"{prefix}*", "*.nii.gz"))
        for f in files:
            new_name = os.path.basename(f)
            new_name = new_name[len(prefix):]
            new_name = new_name[: -len(suffix)] + ".nii.gz"
            to_name = os.path.join(submission_dir, new_name)
            shutil.copy(f, to_name)

    copy_files("volume-covid19-A-0", "_ct_seg.nii.gz", submission_dir)
    copy_files("COVID-19-AR-", "_ct_seg.nii.gz", submission_dir)
    logging.info(f"predictions copied to {submission_dir}.")


if __name__ == "__main__":
    load_dotenv()

    args = types.SimpleNamespace()  # https://stackoverflow.com/a/41765294
    args.mode = os.getenv('mode')
    args.model_folder = os.getenv('model_folder')
    args.data_folder = os.getenv('data_folder')
    args.cache_dir = os.getenv('cache_dir')
    args.prediction_folder = os.getenv('prediction_folder')
    args.gpu = os.getenv('gpu')
    args.model = os.getenv('model', 'Unet').lower()
    args.demo = os.getenv('demo', '0').lower()
    args.split_version = os.getenv('split_version', '0').lower()
    args.loss = os.getenv('loss', 'dice_ce').lower()
    args.start_weights = os.getenv('start_weights', '')
    args.imn_norm = os.getenv('imn_norm', '0')
    args.lr = float(os.getenv('lr', '0.0001'))

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if args.mode == "train":
        data_folder = args.data_folder or os.path.join(
            "Dataset_TrainValidation", "Train")
        train(data_folder=data_folder, model_folder=args.model_folder,
              cache_dir=args.cache_dir, gpu=args.gpu)
    elif args.mode == "infer":
        data_folder = args.data_folder or os.path.join(
            "Dataset_TrainValidation", "Validation")
        infer(data_folder=data_folder, model_folder=args.model_folder,
              prediction_folder=args.prediction_folder, gpu=args.gpu)
    else:
        raise ValueError("Unknown mode.")
