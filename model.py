from compressai.models import CompressionModel
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.entropy_models import entropy_models
from entropy_model import GaussianMixtureConditional
# from lossless.entropy_model import GaussianMixtureConditional
from compressai.layers import GDN, MaskedConv2d
from torch.nn.modules.loss import _Loss

from utils import conv, deconv, update_registered_buffers
from entropy_model import GaussianConditional_LossLess
from compressai.zoo import cheng2020_attn

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class cheng2020_lossless(CompressionModel):
    def __init__(self, N, K=1, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)
        self.K = K
        M = N
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 6, 2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.gaussian_conditional_lossless = GaussianConditional(None)
        self.mse = nn.MSELoss()

    def L2_norm(self, mean, target):
        C = target.size()[1]
        for i in range(self.K):
            # print(i)
            # print("mean: ", mean[:,i*C:(i+1)*C,:,:].size())
            # print("target: ", target.size())
            if i == 0:
                loss = self.mse(mean[:,i*C:(i+1)*C,:,:], target)
            else:
                loss += self.mse(mean[:,i*C:(i+1)*C,:,:], target)
        return loss / self.K

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        gaussian_params_2 = self.g_s(y_hat)

        scales_hat_2, means_hat_2 = gaussian_params_2.chunk(2, 1)
        # weight_2 = nn.functional.softmax(weight_2,dim=1)
        x_hat = self.gaussian_conditional.quantize(
            x, "noise" if self.training else "dequantize"
        )
        _, x_hat_likelihoods = self.gaussian_conditional_lossless(x,  scales_hat_2, means=means_hat_2)

        L2_loss = self.L2_norm(means_hat, y_hat) + self.L2_norm(means_hat_2, x_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "x_hat": x_hat_likelihoods},
            "L2_loss": L2_loss
        }


class cheng2020_lossless_GMM(cheng2020_lossless):
    def __init__(self, N, K=3, **kwargs):
        super().__init__(N, K=K, **kwargs)
        self.gaussian_conditional = GaussianMixtureConditional(K=K)
        self.gaussian_conditional_lossless = GaussianMixtureConditional(K=K)
        self.K = K

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3*3*K, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 8 // 3, N * 3 * K, 1),
        )
########################################################################################################################
        self.entropy_parameters_2 = nn.Sequential(
            nn.Conv2d(27, 27 * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(27 * 10 // 3, 27 * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(27 * 8 // 3, 27 * 3 * K, 1),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat, weight = gaussian_params.chunk(3, 1)
        weight = torch.reshape(weight,(weight.size(0), self.K, weight.size(1)//self.K, weight.size(2), weight.size(3)))
        # print(weight.size())
        weight = nn.functional.softmax(weight,dim=1)
        weight = torch.reshape(weight,(weight.size(0), weight.size(1)*weight.size(2), weight.size(3), weight.size(4)))
        # print(weight.size())
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat, weights=weight)
        params_2 = self.g_s(y_hat)
        # print("params_2: ",params_2.size())
        gaussian_params_2 = self.entropy_parameters_2(params_2)

        scales_hat_2, means_hat_2, weight_2 = params_2.chunk(3, 1)
        weight_2 = torch.reshape(weight_2,(weight_2.size(0), self.K, weight_2.size(1)//self.K, weight_2.size(2), weight_2.size(3)))
        weight_2 = nn.functional.softmax(weight_2,dim=1)
        weight_2 = torch.reshape(weight_2,(weight_2.size(0), weight_2.size(1)*weight_2.size(2), weight_2.size(3), weight_2.size(4)))
        x_hat = self.gaussian_conditional.quantize(
            x, "noise" if self.training else "dequantize"
        )
        _, x_hat_likelihoods = self.gaussian_conditional_lossless(x,  scales_hat_2, means=means_hat_2, weights=weight_2)

        L2_loss = self.L2_norm(means_hat, y_hat) + self.L2_norm(means_hat_2, x_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods, "x_hat": x_hat_likelihoods},
            "L2_loss": L2_loss
        }
