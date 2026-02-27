import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from nucli_train.nets.builders import (
    OPTIMIZERS,
    NETWORK_REGISTRY,
)


class ResNetwithProj(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNetwithProj, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == "conv1":
                module = nn.Conv2d(
                    1, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True),
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


@NETWORK_REGISTRY.register("resnet_proj")
def convnext_proj(**args_cfg):
    model = ResNetwithProj(**args_cfg)
    return model
