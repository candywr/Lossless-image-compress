# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# nohup python -u train.py --cuda --save --checkpoint ./ckpts/lossless_cheng_GMM/90checkpoint.pth.tar > lossless_cheng_GMM_2.log 2>&1 &


import argparse
import math
import random
import shutil
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models, bmshj2018_hyperprior
from pytorch_msssim import ms_ssim
from model import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from PIL import Image
          
def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.)

class RateLoss(nn.Module):
    """Custom rate loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse', K=3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.K = K

    def forward(self, output, target):
        # Epoch = epoch
        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # for means in output["means"]:
        # out['L2_loss'] = output['L2_loss']
        # if Epoch == 1:
        #     out['loss'] = out["bpp_loss"] + out['L2_loss'] * self.lmbda
        #     print("L2 loss:/n",out['loss'])
        # else:
        out['loss'] = out["bpp_loss"]
        # out['loss'] = out["bpp_loss"]
        return out


class RateLoss_L2(nn.Module):
    """Custom rate loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, type='mse', K=3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.type = type
        self.K = K

    def forward(self, output, target):
        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # for means in output["means"]:
        out['L2_loss'] = output['L2_loss']

        out['loss'] = out["bpp_loss"] + out['L2_loss'] * self.lmbda
        print("L2 loss:/n", out['loss'])
        return out

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.inter = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    # def update_inter(self):
    #     self.inter.append(self.avg.item())


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, type='mse'
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        # change the range of image to [0,255] from [0,1]
        d = torch.round(d * 255).float()
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out_net = model(d)
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        # grad norm to avoid grad explosion
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # aux is used for help entopy coding
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()
        # print the information during training process
        if i % 400 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion, type='mse'):
    model.eval()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            d = torch.round(d * 255).float()
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return bpp_loss.avg


def save_checkpoint(state, is_best, epoch, ckpt_path, filename="checkpoint.pth.tar"):
    if epoch % 30 == 0:
        torch.save(state, filename)
    if is_best:
        torch.save(state, ckpt_path + "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="/home/td/Documents/data/imageneta", help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=400,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=20,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=0.6,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=16,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    type = 'mse'
    args = parse_args(argv)
    ckpt_path = "./ckpts"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )


    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    device = 'cuda'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    # net = MeanScaleHyperprior_Lossless(128,192,K=3)
    net = cheng2020_lossless_GMM(192)
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    milestones = [2]
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    criterion = RateLoss(lmbda=args.lmbda, type=type)
    criterion_L2 = RateLoss_L2(lmbda=args.lmbda, type=type)
    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        if epoch == 0:
            train_one_epoch(
                net,
                criterion_L2,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                type
            )
            loss = test_epoch(epoch, test_dataloader, net, criterion_L2, type)
            lr_scheduler.step()

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
        else:
            train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                type
            )
            loss = test_epoch(epoch, test_dataloader, net, criterion, type)
            lr_scheduler.step()

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                epoch,
                ckpt_path,
                ckpt_path + str(epoch) + "checkpoint.pth.tar"
            )


if __name__ == "__main__":
    main(sys.argv[1:])