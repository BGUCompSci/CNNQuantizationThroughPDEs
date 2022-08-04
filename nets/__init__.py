from dataclasses import dataclass
from typing import Any, Dict, List

from resnet import ResNet, BasicBlock, LayerDef, InvertedResidual, MobileNetClosingBlock
from nets.stable_resnet import StableResNet, StableBlock


@dataclass
class ResNetDef:
    net: Any
    layers: List[LayerDef]


_net_configs: Dict[str, ResNetDef] = {
    'resnet18':
        ResNetDef(net=ResNet,
                  layers=[LayerDef(BasicBlock, 64, 64, 2, 1),
                          LayerDef(BasicBlock, 64, 128, 2, 2),
                          LayerDef(BasicBlock, 128, 256, 2, 2),
                          LayerDef(BasicBlock, 256, 512, 2, 2)]),
    'resnet20':
        ResNetDef(net=ResNet,
                  layers=[LayerDef(BasicBlock, 16, 16, 3, 1),
                          LayerDef(BasicBlock, 16, 32, 3, 2),
                          LayerDef(BasicBlock, 32, 64, 3, 2)]),
    'resnet34':
        ResNetDef(net=ResNet,
                  layers=[LayerDef(BasicBlock, 64, 64, 3, 1),
                          LayerDef(BasicBlock, 64, 128, 4, 2),
                          LayerDef(BasicBlock, 128, 256, 6, 2),
                          LayerDef(BasicBlock, 256, 512, 3, 2)]),
    'resnet56':
        ResNetDef(net=ResNet,
                  layers=[LayerDef(BasicBlock, 16, 16, 9, 1),
                          LayerDef(BasicBlock, 16, 32, 9, 2),
                          LayerDef(BasicBlock, 32, 64, 9, 2)]),
    'mobilenetv2':
        ResNetDef(net=ResNet,
                  #                                  in  out n  s  t
                  layers=[LayerDef(InvertedResidual, 16, 16, 1, 1, 1),
                          LayerDef(InvertedResidual, 16, 24, 2, 1, 6),
                          LayerDef(InvertedResidual, 24, 32, 3, 2, 6),
                          LayerDef(InvertedResidual, 32, 64, 4, 2, 6),
                          LayerDef(InvertedResidual, 64, 96, 3, 1, 6),
                          LayerDef(InvertedResidual, 96, 160, 3, 2, 6),
                          LayerDef(InvertedResidual, 160, 320, 1, 1, 6),
                          LayerDef(MobileNetClosingBlock, 320, 1280, 1, 1)]),
    'stablenet20':
        ResNetDef(net=StableResNet,
                  layers=[LayerDef(StableBlock, 16, 16, 3, 1),
                          LayerDef(StableBlock, 16, 32, 3, 2),
                          LayerDef(StableBlock, 32, 64, 3, 2)]),
    'stablenet20x2':
        ResNetDef(net=StableResNet,
                  layers=[LayerDef(StableBlock, 16, 16, 6, 1),
                          LayerDef(StableBlock, 16, 32, 6, 2),
                          LayerDef(StableBlock, 32, 64, 6, 2)]),
    'stablenet34':
        ResNetDef(net=StableResNet,
                  layers=[LayerDef(StableBlock, 64, 64, 3, 1),
                          LayerDef(StableBlock, 64, 128, 4, 2),
                          LayerDef(StableBlock, 128, 256, 6, 2),
                          LayerDef(StableBlock, 256, 512, 3, 2)])
    ,
    'stablenet56':
        ResNetDef(net=StableResNet,
                  layers=[LayerDef(StableBlock, 16, 16, 9, 1),
                          LayerDef(StableBlock, 16, 32, 9, 2),
                          LayerDef(StableBlock, 32, 64, 9, 2)]),
    'stablenet56x2':
        ResNetDef(net=StableResNet,
                  layers=[LayerDef(StableBlock, 16, 16, 18, 1),
                          LayerDef(StableBlock, 16, 32, 18, 2),
                          LayerDef(StableBlock, 32, 64, 18, 2)]),
}


def make_net(arch, classes, **kwargs):
    if arch not in _net_configs:
        raise Exception(f'Arch {arch} not supported by StableNet module')

    config: ResNetDef = _net_configs[arch]
    return config.net(classes, config.layers, **kwargs)
