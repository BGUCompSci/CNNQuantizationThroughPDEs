from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
from torch import nn as nn
from torch.nn import functional as F

from net_components.quantization import \
    apot_quantization, uq_with_calibrated_gradients, uniform_quantization, build_power_value


@dataclass
class LayerDef:
    block: Any
    in_channels: int
    out_channels: int
    num_blocks: int
    stride: int = 1
    expansion: int = 1


class ResNet(nn.Module):
    def __init__(self, classes: int, layers: List[LayerDef], in_channels: int = 3,
                 stable: bool = False, stability_coeff: float = 1.0, groups: int = 1,
                 quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                 return_layer_activations: Union[bool, str] = False):
        super(ResNet, self).__init__()

        self.stable = stable
        self.stability_coeff = stability_coeff
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.return_layer_activations = return_layer_activations
        self.groups = groups

        self.conv1 = conv2d_weight_parameter(in_channels, layers[0].in_channels)
        self.bn1 = nn.BatchNorm2d(layers[0].in_channels)

        self.layers = nn.Sequential(*[self._make_layer(layer) for layer in layers])

        self.dense_output = nn.Linear(layers[-1].out_channels * layers[-1].expansion, classes)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def set_quantize_weights(self, quantize_weights: int) -> None:
        if not self.quantize_weights and quantize_weights:
            raise ValueError('Cant change architecture from unquantized weights to quantized')

        if self.quantize_weights and not quantize_weights:
            raise ValueError('Cant change architecture from quantized weights to unquantized')

        self.quantize_weights = quantize_weights

        for layer in self.layers:
            for module in layer.modules():
                if hasattr(module, 'set_quantize_weights'):
                    module.set_quantize_weights(quantize_weights)

    def set_quantize_activations(self, quantize_activations: int) -> None:
        if not self.quantize_activations and quantize_activations:
            raise ValueError('Cant change architecture from unquantized activations to quantized')

        if self.quantize_activations and not quantize_activations:
            raise ValueError('Cant change architecture from quantized activations to unquantized')

        self.quantize_activations = quantize_activations

        for layer in self.layers:
            for module in layer.modules():
                if hasattr(module, 'set_quantize_activations'):
                    module.set_quantize_activations(quantize_activations)

    def _make_layer(self, layer: LayerDef) -> nn.Module:
        layers = [layer.block(layer.in_channels, layer.out_channels, layer.expansion, False,
                              quantize_weights=self.quantize_weights, quantize_activations=self.quantize_activations,
                              stride=layer.stride, groups=self.groups)]
        for _ in range(1, layer.num_blocks):
            layers.append(layer.block(layer.out_channels, layer.out_channels, layer.expansion,
                                      self.stable, self.stability_coeff,
                                      self.quantize_weights, self.quantize_activations,
                                      groups=self.groups))

        return nn.Sequential(*layers)

    def forward(self, data: torch.Tensor):
        layer_activations = []

        out = F.conv2d(data, self.conv1, padding=1)
        out = self.bn1(out)
        out = F.relu(out, True)

        for layer in self.layers:
            if type(self.return_layer_activations) is bool:
                out = layer(out)
                layer_activations.append(out)
            elif self.return_layer_activations == 'all':
                for block in layer:
                    out = block(out)
                    layer_activations.append(out)

        out = F.avg_pool2d(out, out.size(-1))

        out = out.view(out.size(0), -1)
        out = self.dense_output(out)

        if self.return_layer_activations:
            return out, layer_activations
        else:
            return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1,
                 stable: bool = False, stability_coeff: float = 1.0,
                 quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                 stride: int = 1, groups: int = 1):
        super(BasicBlock, self).__init__()
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if expansion != 1:
            raise ValueError('BasicBlock only supports expansion=1')

        self.stable = stable
        self.stability_coeff = stability_coeff if stable else 1
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.stride = stride

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample_weight = conv2d_weight_parameter(in_channels, out_channels * self.expansion, kernel_size=1)
            self.downsample_bn = nn.BatchNorm2d(out_channels * self.expansion)
            if quantize_weights:
                self.downsample_quantizer = Quantizer(bits=quantize_weights)
            else:
                self.downsample_quantizer = nn.Identity()
        else:
            self.downsample_weight = None
            self.downsample_bn = None
            self.downsample_quantizer = None

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_weight = conv2d_weight_parameter(in_channels, out_channels)
        if quantize_weights:
            self.conv1_quantizer = Quantizer(bits=quantize_weights)
        else:
            self.conv1_quantizer = nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)

        if not stable:
            self.conv2_weight = conv2d_weight_parameter(out_channels, out_channels)
            if quantize_weights:
                self.conv2_quantizer = Quantizer(bits=quantize_weights)
            else:
                self.conv2_quantizer = nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels)

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

            if self.downsample_quantizer:
                self.downsample_quantizer.set_bits(bits)

    def set_quantize_activations(self, bits: int) -> None:
        self.quantize_activations = bits

        self.act1_quantizer.set_bits(bits)
        self.act2_quantizer.set_bits(bits)

    def forward(self, data: torch.Tensor):
        data = self.act1_quantizer(data)

        conv1_weight = self.conv1_quantizer(self.conv1_weight)
        out = F.conv2d(data, conv1_weight, stride=self.stride, padding=1)
        out = self.bn1(out)
        out = F.relu(out, True)
        out = self.act2_quantizer(out)

        if self.stable:
            conv1_weight = self.conv1_quantizer(self.conv1_weight)
            out = F.conv_transpose2d(out, conv1_weight, stride=self.stride, padding=1)
        else:
            conv2_weight = self.conv2_quantizer(self.conv2_weight)
            out = F.conv2d(out, conv2_weight, padding=1)
        out = self.bn2(out)

        if self.downsample_weight is None:
            residual = data
        else:
            downsample_weight = self.downsample_quantizer(self.downsample_weight)
            residual = F.conv2d(data, downsample_weight, stride=self.stride)
            residual = self.downsample_bn(residual)

        out = residual - self.stability_coeff * out
        out = F.relu(out, True)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expansion: int = 1,
                 stable: bool = False, stability_coeff: float = 1.0,
                 quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                 stride: int = 1, groups: int = 1):
        super(InvertedResidual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.stable = stable
        self.stability_coeff = stability_coeff if stable else 1
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations
        self.stride = stride

        self.hidden_channels = round(in_channels * expansion)

        self.conv1_weight = conv2d_weight_parameter(in_channels, self.hidden_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.hidden_channels)
        if quantize_weights:
            self.conv1_quantizer = Quantizer(bits=quantize_weights)
        else:
            self.conv1_quantizer = nn.Identity()

        # groups=self.hidden_dim means separable depth-wise conv
        self.conv2_weight = conv2d_weight_parameter(self.hidden_channels, self.hidden_channels,
                                                    groups=self.hidden_channels, kernel_size=3)
        if quantize_weights:
            self.conv2_quantizer = Quantizer(bits=8)
        else:
            self.conv2_quantizer = nn.Identity()

        if stable:
            self.bn2 = nn.BatchNorm2d(in_channels)
        else:
            self.bn2 = nn.BatchNorm2d(self.hidden_channels)

            self.conv3_weight = conv2d_weight_parameter(self.hidden_channels, out_channels, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(out_channels)
            if quantize_weights:
                self.conv3_quantizer = Quantizer(bits=quantize_weights)
            else:
                self.conv3_quantizer = nn.Identity()

        if quantize_activations:
            self.act1_quantizer = Quantizer(is_weight=False, bits=quantize_activations)
            self.act2_quantizer = Quantizer(is_weight=False, bits=quantize_activations)
            if not stable:
                self.act3_quantizer = Quantizer(is_weight=False, bits=quantize_activations)
        else:
            self.act1_quantizer = nn.Identity()
            self.act2_quantizer = nn.Identity()
            if not stable:
                self.act3_quantizer = nn.Identity()

    def set_quantize_weights(self, bits: int) -> None:
        if self.quantize_weights:
            self.quantize_weights = bits

            self.conv1_quantizer.set_bits(bits)
            self.conv2_quantizer.set_bits(bits)

            if not self.stable:
                self.conv3_quantizer.set_bits(bits)

    def set_quantize_activations(self, bits: int) -> None:
        if self.quantize_activations:
            self.quantize_activations = bits

            self.act1_quantizer.set_bits(bits)
            self.act2_quantizer.set_bits(bits)
            if not self.stable:
                self.act3_quantizer.set_bits(bits)

    def forward(self, data: torch.Tensor):
        data = self.act1_quantizer(data)

        conv1_weight = self.conv1_quantizer(self.conv1_weight)
        out = F.conv2d(data, conv1_weight)

        if self.stable:
            conv2_weight = self.conv2_quantizer(self.conv2_weight)
            out = F.conv2d(out, conv2_weight, groups=self.hidden_channels)

            out = self.bn1(out)
            out = F.relu(out, True)
            out = self.act2_quantizer(out)

            out = F.conv_transpose2d(out, conv2_weight, groups=self.hidden_channels)
            out = F.conv_transpose2d(out, conv1_weight)
            out = self.bn2(out)

            out = data - self.stability_coeff * out
            out = torch.cat([out, data[:, 0:(self.out_channels - self.in_channels), :, :]], dim=1)
            if self.stride == 2:
                out = F.avg_pool2d(out, 2)
        else:
            out = self.bn1(out)
            out = F.relu(out, True)
            out = self.act2_quantizer(out)

            conv2_weight = self.conv2_quantizer(self.conv2_weight)
            out = F.conv2d(out, conv2_weight, stride=self.stride, padding=1, groups=self.hidden_channels)
            out = self.bn2(out)
            out = F.relu(out, True)
            out = self.act3_quantizer(out)

            conv3_weight = self.conv3_quantizer(self.conv3_weight)
            out = F.conv2d(out, conv3_weight)
            out = self.bn3(out)

            if self.stride == 1 and self.in_channels == self.out_channels:
                out = data - self.stability_coeff * out

        return out


def MobileNetClosingBlock(in_channels: int, out_channels: int, expansion: int = 1,
                          stable: bool = False,
                          quantize_weights: Optional[int] = None, quantize_activations: Optional[int] = None,
                          stride: int = 1, groups: int = 1, **kwargs) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, groups=groups, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))


def conv2d_weight_parameter(in_channels: int, out_channels: int, kernel_size: int = 3, groups: int = 1) -> nn.Parameter:
    weight = nn.Parameter(torch.zeros((out_channels, in_channels // groups, kernel_size, kernel_size)))
    nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')

    return weight


class Quantizer(nn.Module):
    def __init__(self, is_weight: bool = True, bits: int = 5,
                 power: bool = False, additive: bool = False,
                 grad_scale: Optional[float] = None, use_weight_bias: bool = False):
        super(Quantizer, self).__init__()

        self.is_weight = is_weight
        self.bits = bits
        self.power = power
        self.additive = additive
        self.grad_scale = grad_scale
        self.use_weight_bias = use_weight_bias
        self.layer_type = 'Quantizer'

        self.init_proj_set(additive, bits, is_weight, power)

        init_val = 3.0 if is_weight else 6.0
        self.alpha = torch.nn.Parameter(torch.tensor(init_val))

        if is_weight and use_weight_bias:
            self.quantized_weight_bias = torch.nn.Parameter(0.01 * torch.randn(1))

    def set_bits(self, bits: int) -> None:
        self.bits = bits
        self.init_proj_set(self.additive, self.bits, self.is_weight, self.power)

    def init_proj_set(self, additive: bool, bits: int, is_weight: bool, power: bool) -> None:
        if power:
            if bits > 2 and is_weight:
                self.proj_set = build_power_value(B=bits - 1, additive=additive)
            else:
                self.proj_set = build_power_value(B=bits, additive=additive)

    def forward(self, tensor: torch.Tensor):
        if self.is_weight:
            mean = tensor.mean()
            std = tensor.std()
            tensor = tensor.add(-mean).div(std)

        if self.bits <= 2 and self.is_weight:
            return uq_with_calibrated_gradients(self.grad_scale)(tensor, self.alpha)
        else:
            if self.power:
                return apot_quantization(tensor, self.alpha, self.proj_set, self.is_weight, self.grad_scale)
            else:
                bias = self.quantized_weight_bias if self.is_weight and self.use_weight_bias else None
                return uniform_quantization(tensor, self.alpha, self.bits, self.is_weight, self.grad_scale, bias)
