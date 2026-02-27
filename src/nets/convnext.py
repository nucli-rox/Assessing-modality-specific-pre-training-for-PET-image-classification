# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root above scripts/
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sparse.sparse_transform import SparseConvNeXtBlock, SparseConvNeXtLayerNorm, SparseConv2dReweighted
import src.sparse.sparse_transform as sparse_transform
from nucli_train.nets.builders import (
    OPTIMIZERS,
    NETWORK_REGISTRY,
)
import math
from nucli_train.nets.conv_blocks import CONV_BLOCKS_REGISTRY

class SparseConvNeXt_2d(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=1,
        decoder_block=None,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 192, 384],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        decoder_embed_dim=512,
        decoder_depth=1,
        learnable_mask_token=None,
        patch_size=32,
        mask_ratio=0.6,
        prepretrained_ckpt=None,
        sparse=True,
        reweighted=False,
        **kwargs,
    ):
        super().__init__()

        self.dims = dims
        self.depths = depths
        self.in_chans = in_chans
        self.sparse = sparse
        self.reweighted = reweighted
        self.prepretrained_ckpt = prepretrained_ckpt
        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            SparseConv2dReweighted(in_chans, self.dims[0], kernel_size=4, stride=4),
            SparseConvNeXtLayerNorm(self.dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(len(self.dims) - 1):
            downsample_layer = nn.Sequential(
                SparseConvNeXtLayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                SparseConv2dReweighted(self.dims[i], self.dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        self.out_chans = in_chans
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(len(self.dims)):
            stage = nn.Sequential(
                *[
                    SparseConvNeXtBlock(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        sparse=self.sparse,
                        reweighted=self.reweighted,
                    )
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        self.num_stages = len(self.stages)
        if num_classes > 0:
            self.norm = SparseConvNeXtLayerNorm(
                self.dims[-1], eps=1e-6, sparse=False
            )  # final norm layer for LE/FT; should not be sparse
            self.fc = nn.Linear(self.dims[-1], num_classes)
        else:
            self.norm = nn.Identity()  # final norm layer
            self.fc = nn.Identity()

        self.apply(self._init_weights)
        self.opt = None

        # decoder
        decoder_cls = CONV_BLOCKS_REGISTRY.get(decoder_block)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.DecoderBlock = decoder_cls
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.proj = nn.Conv2d(
            in_channels=dims[-1], out_channels=decoder_embed_dim, kernel_size=1
        )

        if learnable_mask_token:
            self.mask_token = nn.Parameter(
                torch.zeros(1, decoder_embed_dim, 1, 1)
            )  # Each channel has its own mask token
        else:
            self.mask_token = torch.zeros(
                1, decoder_embed_dim, 1, 1, requires_grad=False
            ).cuda()

        decoder = [
            self.DecoderBlock(dim=decoder_embed_dim, drop_path=0.0)
            for i in range(decoder_depth)
        ]
        self.decoder = nn.Sequential(*decoder)
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=self.patch_size**2 * in_chans,
            kernel_size=1,
        )

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        B, _, h, w = mask.shape
        mask = mask.view(B, 1, h, w).type_as(x)
        mask = mask.expand(B, x.shape[1], h, w)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * mask + mask_token * (1 - mask)

        x = self.decoder(x)

        pred = self.pred(x)
        return pred

    def set_optimizer(self, opt_name, opt_args):
        self.opt = OPTIMIZERS[opt_name](self.parameters(), **opt_args)

    def get_optimizer(self):
        k = 1 / 16
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=math.sqrt(k) * 1e-4
        )
        return optimizer

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """
        Forward pass of the Sparse ConvNeXtV2 model.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            mask (torch.Tensor): Mask tensor of shape [B, N] where N is
                the vectorization of a pxpxp patch.
        Returns:
            torch.Tensor: Output tensor of shape [B, C, D, H, W].
        The usage of sparse tensors is done in the intermediate layers.
        """

        self.original_shape = list(x.shape)
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)

            if i > 0:
                self.original_shape[1] = self.dims[i]
                self.original_shape[2] = self.original_shape[2] // 2
                self.original_shape[3] = self.original_shape[3] // 2
        return x

    def upsample_mask(self, mask, h, w, scale_h, scale_w):
        mask = mask.reshape(-1, h, w)
        mask = mask.repeat_interleave(scale_h, dim=1)
        mask = mask.repeat_interleave(scale_w, dim=2)
        return mask

    def forward(self, x, mask):
        """
        Forward pass of the Sparse ConvNeXtV2 model.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W].
            mask (torch.Tensor): Mask tensor of shape [B, N] where N is
                the vectorization of a pxpxp patch.
        Returns:
            torch.Tensor: Output tensor of shape [B, C, D, H, W].
        The usage of sparse tensors is done in the intermediate layers.
        """
        mask_dec = mask.clone()

        self.original_shape = list(x.shape)
        # print(f"Stage 0, shape {x.shape}")
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
            print("mean of x during stage", i, ":", x.mean())

            if i > 0:
                ### We update the shape
                self.original_shape[1] = self.dims[i]
                self.original_shape[2] = self.original_shape[2] // 2
                self.original_shape[3] = self.original_shape[3] // 2
            # print(f"Stage {i}, shape {x.shape}")
        feats = x.clone()
        x = self.forward_decoder(x, mask_dec)

        return x, feats

@NETWORK_REGISTRY.register("sparseconvnext_2d")
def sparseconvnext_2d(**args_cfg):
    model = SparseConvNeXt_2d(**args_cfg)
    return model

