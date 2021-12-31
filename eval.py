import torch
import time
from compressai.datasets import ImageFolder
from compressai.zoo import models, bmshj2018_hyperprior
import warnings
import cv2
import torch
import numpy as np
from pytorch_msssim import ms_ssim
import torch.nn as nn
import os
import sys
import torch.nn.functional as F
import math
import argparse
import random
import time
# pylint: disable=E0611,E0401
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
import shutil
from PIL import Image
from model import *
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.zoo import models
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
# def eval():
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.is_available())
def compute_psnr(a, b):
    mse = torch.mean((a/255 - b/255)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[1] * size[2] * size[3]
    res = 0
    for n, likelihoods in out_net['likelihoods'].items():
        print(n)
        print((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)))
        res += (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)).item()
    return res

def pad(x, new_h=2 ** 11, new_w=2**11):
    h, w = x.size(2), x.size(3)
    H = (h + new_h - 1) // new_h * new_h
    W = (w + new_w - 1) // new_w * new_w
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size[0], size[1]
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=500,
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
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
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
        default=(256, 256),
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
    # parser.add_argument("--checkpoint", type=str, default='./ckpts/lossless_cheng_singleG/210checkpoint.pth.tar', help="Path to a checkpoint")
    # parser.add_argument("--checkpoint", type=str, default='./ckpts/lossless_cheng_singleG/checkpoint_best_loss.pth.tar', help="Path to a checkpoint")
    parser.add_argument("--checkpoint", type=str, default='./ckpts/lossless_cheng_GMM/checkpoint_best_loss.pth.tar', help="Path to a checkpoint")
    parser.add_argument(
        "-p", "--path", type=str, default="/home/liu/dataset/kodak/"
    )
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)

    # path = "./dataset/test"
    path = args.path
    # path = "/home/liu/dataset/CLIC/test/"
    # path = "/home/liu/dataset/CLIC/Mobile_test/mobile/"
    # path = "/home/liu/dataset/CLIC/Professional_test/professional/"
    # path = "/home/liu/dataset/CLIC/Professional_val/valid/"
    # path = "/home/liu/dataset/CLIC/Mobile_val/valid/"
    # path = ""
    output_path = "./res/"
    img_list = []
    for file in os.listdir(path):
        if file[-3:] in ["jpg", "png", "peg"]:
            img_list.append(file)
    device = 'cuda'
    # device = 'cpu'

    # net = cheng2020_lossless(192)
    net = cheng2020_lossless_GMM(192)
    net = net.to(device)
    net.eval()
    count = 0
    PSNR = 0
    Bit_rate = 0
    SSIM = 0
    total_time = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    # count = 0
    for img_name in img_list:
        # count += 1
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        x = x*255
        x = torch.round(x).float()

        size = x.size()
        h, w = size[2], size[3]
        print(x.size())
        if h % 64 == 0:
            new_h = h
        else:
            new_h = h - h % 64 + 64
        if w % 64 == 0:
            new_w = w
        else:  
            new_w = w - w % 64 + 64
        x = pad(x, new_h, new_w)
        print("padded size", x.size())
        count += 1
        with torch.no_grad():
            # torch.cuda.synchronize()
            s = time.time()
            out_net = net.forward(x)
            # torch.cuda.synchronize()
            e = time.time()
            if count > 1:
                total_time += (e - s)
            out_net['x_hat'].clamp_(0, 255)
            out_net['x_hat'] = crop(out_net['x_hat'], (size[2],size[3]))
            x = crop(x, (size[2],size[3]))
            # print(f'PSNR: {compute_psnr(x, out_net["x_hat"]):.2f}dB')
            print(f'Bit-rate: {compute_bpp(out_net):.3f} bpp')
            # PSNR += compute_psnr(x, out_net["x_hat"])
            Bit_rate += compute_bpp(out_net)

        # image = x.detach().cpu().numpy()[0]
        # res = crop(out_net["x_hat"], (size[2],size[3]))
        # res = res.detach().cpu().numpy()[0]
        # image = image.astype("uint8")
        # res = res.astype("uint8")
        # plt.figure(figsize=(12,12))
        # plt.subplot(1,2,1)
        # plt.imshow(np.transpose(image, (1,2,0)))
        # plt.subplot(1,2,2)
        # plt.imshow(np.transpose(res,(1,2,0)))
        # plt.savefig(output_path + str(count) + ".png")
        # # break
    PSNR = PSNR / count
    Bit_rate = Bit_rate / count
    total_time = total_time / (count - 1)
    print(f'average_PSNR: {PSNR:.2f}dB')
    print(f'average_Bit-rate: {Bit_rate:.3f} bpp')
    print(f'average_time: {total_time:.3f} ms')
    

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main(sys.argv[1:])
    