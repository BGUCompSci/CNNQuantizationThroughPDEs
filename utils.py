import json
import logging.config
import os
import re
import socket
from functools import cache
from typing import List, Dict, Any, Union, OrderedDict

import torch
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent
from torch import Tensor
from torch.backends import cudnn as cudnn
from torch.nn import functional as F

from config import CONFIG
from nets import make_net
from datasets import load_dataset

CONFIG_FILE_NAME = 'config.json'
SUMMARY_FILE_NAME = 'summary_metrics.json'


def activation_loss(device: torch.device, lowp_bits: float,
                    highp_activations: List[torch.Tensor],
                    lowp_activations: List[torch.Tensor]) -> torch.Tensor:
    loss = torch.as_tensor(0.0, device=device)

    for ha, la in zip(highp_activations, lowp_activations):
        shrink_coeff = (torch.max(torch.abs(ha)) / (2 ** lowp_bits - 1)).detach()
        loss += torch.norm(F.softshrink(ha - la, shrink_coeff.item())) ** 2

    return loss * (1.0 / len(highp_activations))


def test_net(net: torch.nn.Module, device: torch.device, data_loader: torch.utils.data.DataLoader) -> float:
    correct = 0.0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if len(outputs) == 2:
                outputs, _ = outputs

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    return correct / len(data_loader.dataset)


def load_net_from(config, classes: int, device,
                  return_layer_activations: Union[bool, str] = True, transform_net: bool = False):
    results_dir = config['results_dir']
    name = config['net']
    net_filename = os.path.join(results_dir, name + '.pth')
    state_dict = torch.load(net_filename, map_location=device)

    if transform_net:
        new_state_dict = {}
        for key in state_dict.keys():
            if 'act3' in key:
                continue
            if re.search(r'\.conv\d$', key) and '_weight' not in key:
                new_state_dict[key + '_weight'] = state_dict[key]
            else:
                new_state_dict[key] = state_dict[key]
    else:
        new_state_dict = state_dict

    net = make_net(arch=name, classes=classes,
                   stable=config['stable'],
                   stability_coeff=config['stability_coeff'],
                   quantize_weights=config['quantize_weights'],
                   quantize_activations=config['quantize_activations'],
                   return_layer_activations=return_layer_activations).to(device)

    net.load_state_dict(new_state_dict)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    return net


def load_test_dataset_for(config) -> torch.utils.data.DataLoader:
    dataset = config['dataset']
    dataset_dir = config['dataset_dir']
    batch_size = config['batch_size']
    train_part, val_part = config['train_val_split']
    num_workers = config['num_workers']

    return load_test_dataset(dataset, dataset_dir, batch_size, train_part, val_part, num_workers)


@cache
def load_test_dataset(dataset, dataset_dir, batch_size, train_part, val_part, num_workers):
    dataset_dir = os.path.join(dataset_dir, dataset)
    _, _, test_loader, classes = load_dataset(dataset, batch_size, train_part, val_part,
                                              num_workers, data_root=dataset_dir, test_only=True)
    return test_loader, classes


def load_experiment(experiment_path: str) -> Dict:
    config_file = os.path.join(experiment_path, CONFIG_FILE_NAME)
    if not os.path.isfile(config_file):
        raise ValueError(f'Invalid experiment specification {experiment_path}')

    with open(config_file, 'r') as f:
        return json.load(f)


def save_experiment(config: Dict, experiment_path: str) -> None:
    config_file = os.path.join(experiment_path, CONFIG_FILE_NAME)
    os.makedirs(experiment_path, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f)


def update_global_config(cmd_line_opts):
    if 'experiment_label' not in cmd_line_opts:
        raise ValueError('experiment_label must be set!')

    CONFIG.update(cmd_line_opts)

    if 'results_dir' not in cmd_line_opts:
        CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'], CONFIG['net_type'],
                                                   CONFIG['net'], CONFIG['experiment_type'],
                                                   CONFIG['dataset'], CONFIG['experiment_label'])


def load_summary(experiment_path: str) -> Dict[str, Any]:
    summary_file = os.path.join(experiment_path, SUMMARY_FILE_NAME)
    return json.load(open(summary_file, 'r'))


def save_summary(summary: Dict[str, Any], experiment_path: str) -> None:
    summary_file = os.path.join(experiment_path, SUMMARY_FILE_NAME)
    os.makedirs(experiment_path, exist_ok=True)
    json.dump(summary, open(summary_file, 'w'))


def init_wandb():
    wandb.init(project=CONFIG['wandb_project_name'], entity=CONFIG['wandb_entity'],
               name=CONFIG['experiment_label'], sync_tensorboard=True,
               config=CONFIG,
               config_exclude_keys=['pretrained_path', 'imagenet',
                                    'gil_imagenet_debug', 'cityscapes',
                                    'save_statistics', 'retrain', 'save_model',
                                    'get_stats', 'pretrained', 'activation_num',
                                    'num_gpu', 'CUDA_VISIBLE_DEVICES'])


def upload_summary_to_wandb(mode, summary, w):
    for run in w.runs(mode['wandb_project_name']):
        if run.name == mode['experiment_label']:
            run.summary['test/acc'] = summary['test_acc']
            run.summary.update()


def load_tb_metrics(experiment_path: str) -> (List[float], List[float], List[float], List[float]):
    event_acc = EventAccumulator(experiment_path)
    event_acc.Reload()

    train_accs = extract_values(event_acc.Scalars('train/acc'))
    train_losses = extract_values(event_acc.Scalars('train/loss'))
    val_accs = extract_values(event_acc.Scalars('val/acc'))
    val_losses = extract_values(event_acc.Scalars('val/loss'))

    return train_accs, train_losses, val_accs, val_losses,


def transform_state_dict(state_dict: OrderedDict[str, Tensor], device: torch.device) -> OrderedDict[str, Tensor]:
    # Remove module if no DataParallel
    if device != 'cuda':
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    return state_dict


def glob_dirs(glob_pattern: str):
    for dir_path, dir_names, _ in os.walk(glob_pattern):
        if not dir_names:
            yield dir_path


def extract_values(series: List[ScalarEvent]) -> List[float]:
    return [t.value for t in series]


def init_logger(log_file: str):
    logging.config.dictConfig(get_logger_config_dic(log_file))


def get_logger_config_dic(log_file: str):
    log_dir = os.path.dirname(os.path.abspath(log_file))
    os.makedirs(log_dir, exist_ok=True)

    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'verbose': {
                'format': "%(asctime)s.%(msecs)03d; %(levelname)s; [%(name)s %(lineno)s]; {}; %(message)s".format(
                    socket.gethostname()),
                'datefmt': "%Y-%m-%d %H:%M:%S",
            },
            'simple': {
                'format': '%(asctime)s.%(msecs)03d; %(levelname)s; [%(name)s %(lineno)s]; %(message)s',
                'datefmt': "%Y-%m-%d %H:%M:%S",
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple'
            },
            'rotating_file': {
                'level': 'INFO',
                'formatter': 'verbose',
                'class': 'logging.FileHandler',
                'filename': log_file
            },
            'rotating_file_debug': {
                'level': 'DEBUG',
                'formatter': 'verbose',
                'class': 'logging.FileHandler',
                'filename': f"{log_file}.debug"
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["console", "rotating_file", "rotating_file_debug"]
        },
        "loggers": {
            "PIL": {
                "level": "CRITICAL",
                "handlers": ["console", "rotating_file", "rotating_file_debug"],
                "propagate": False
            }
        }
    }
