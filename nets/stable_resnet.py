from typing import List, Optional

import torch
from torch import nn as nn
from torch.nn import functional as F

from nets.resnet import ResNet, conv2d_weight_parameter, Quantizer, LayerDef


class StableResNet(ResNet):
    def __init__(self, classes: int, layers: List[LayerDef], in_channels: int = 3,
                 stable: bool = False, stability_coeff: float = 1.0, groups: int = 1,
                 quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                 return_layer_activations: bool = False):
        super(StableResNet, self).__init__(classes, layers, in_channels,
                                           stable, stability_coeff, groups,
                                           quantize_weights, quantize_activations,
                                           return_layer_activations)

    def _make_layer(self, layer: LayerDef) -> nn.Module:
        layers = [layer.block(layer.in_channels, layer.out_channels, layer.expansion, layer.stride == 2,
                              self.stable, self.stability_coeff,
                              self.quantize_weights, self.quantize_activations,
                              groups=self.groups)]
        for _ in range(1, layer.num_blocks):
            layers.append(layer.block(layer.out_channels, layer.out_channels, layer.expansion, False,
                                      self.stable, self.stability_coeff,
                                      self.quantize_weights, self.quantize_activations,
                                      groups=self.groups))

        return nn.Sequential(*layers)


class StableBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1,
                 pool: bool = False,
                 stable: bool = False, stability_coeff: float = 1.0,
                 quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                 stride: int = 1, groups: int = 1):
        super(StableBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if expansion != 1:
            raise ValueError('BasicBlock only supports expansion=1')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool = pool
        self.stable = stable
        self.stability_coeff = stability_coeff if stable else 1
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_weight = conv2d_weight_parameter(in_channels, in_channels)
        if quantize_weights:
            self.conv1_quantizer = Quantizer(bits=quantize_weights)
        else:
            self.conv1_quantizer = nn.Identity()
        self.bn1 = nn.BatchNorm2d(in_channels)

        if not stable:
            self.conv2_weight = conv2d_weight_parameter(in_channels, in_channels)
            if quantize_weights:
                self.conv2_quantizer = Quantizer(bits=quantize_weights)
            else:
                self.conv2_quantizer = nn.Identity()
        self.bn2 = nn.BatchNorm2d(in_channels)

        if quantize_activations:
            self.act1_quantizer = Quantizer(is_weight=False, bits=quantize_activations)
            self.act2_quantizer = Quantizer(is_weight=False, bits=quantize_activations)
        else:
            self.act1_quantizer = nn.Identity()
            self.act2_quantizer = nn.Identity()

    def set_quantize_weights(self, bits: int) -> None:
        if self.quantize_weights:
            self.quantize_weights = bits

            self.conv1_quantizer.set_bits(bits)

            if not self.stable:
                self.conv2_quantizer.set_bits(bits)

    def set_quantize_activations(self, bits: int) -> None:
        self.quantize_activations = bits

        self.act1_quantizer.set_bits(bits)
        self.act2_quantizer.set_bits(bits)

    def forward(self, data: torch.Tensor):
        data = self.act1_quantizer(data)

        conv1_weight = self.conv1_quantizer(self.conv1_weight)
        out = F.conv2d(data, conv1_weight, padding=1)
        out = self.bn1(out)
        out = F.relu(out, True)
        out = self.act2_quantizer(out)

        if self.stable:
            conv1_weight = self.conv1_quantizer(self.conv1_weight)
            out = F.conv_transpose2d(out, conv1_weight, padding=1)
        else:
            conv2_weight = self.conv2_quantizer(self.conv2_weight)
            out = F.conv2d(out, conv2_weight, padding=1)
        out = self.bn2(out)

        out = data - self.stability_coeff * out
        out = F.relu(out, True)

        out = torch.cat([out, data[:, 0:(self.out_channels - self.in_channels), :, :]], dim=1)
        if self.pool:
            out = F.avg_pool2d(out, 2)

        return out
