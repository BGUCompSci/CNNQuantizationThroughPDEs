import csv
import logging
import os
import pprint

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.backends import cudnn as cudnn
from torch.nn import functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG
from nets import make_net
from datasets import load_dataset, dataset_classes
from utils import activation_loss, update_global_config, test_net, save_summary, load_experiment, \
    save_experiment, glob_dirs, load_test_dataset_for, load_net_from, init_logger, init_wandb

logger = logging.getLogger(__name__)


def train_single(config) -> None:
    name = config['net']
    dataset = config['dataset']
    dataset_dir = config['dataset_dir']
    batch_size = config['batch_size']
    train_part, val_part = config['train_val_split']
    num_workers = config['num_workers']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = config['results_dir']
    net_filename = os.path.join(results_dir, name + '.pth')
    criterion = F.cross_entropy
    lr = config['lr']
    weight_decay = config['weight_decay']
    epoch_num = config['epoch_num']

    stable = config['stable']
    stability_coeff = config['stability_coeff']
    quantize_weights_bits = config['quantize_weights']
    quantize_activations_bits = config['quantize_activations']
    weight_effective_bits = 16
    act_effective_bits = 16
    weight_bits = None if not quantize_weights_bits else weight_effective_bits
    act_bits = None if not quantize_activations_bits else act_effective_bits

    # Prepare dataset
    train_loader, validate_loader, test_loader, classes = \
        load_dataset(dataset, batch_size, train_part, val_part, num_workers,
                     data_root=os.path.join(dataset_dir, dataset))

    # Prepare network
    net = make_net(arch=name, classes=classes,
                   stable=stable, stability_coeff=stability_coeff,
                   quantize_weights=weight_bits, quantize_activations=act_bits).to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    wandb.watch(net, log='all', log_freq=len(train_loader.dataset) // batch_size)

    # Prepare training instruments
    optimizer = SGD(net.parameters(), lr, 0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
    tensorboard = SummaryWriter(results_dir)

    best_val_acc = 0
    for epoch in range(epoch_num):
        logger.info(f'Epoch {epoch + 1}/{config["epoch_num"]}')

        # Train epoch and log stats
        train_loss, train_acc = train_step(net, optimizer, criterion, train_loader, device)
        effective_lr = optimizer.param_groups[0]['lr']
        logger.info(f'LR: {effective_lr}')
        logger.info(f'Train acc = {train_acc}')
        tensorboard.add_scalar('lr', effective_lr, epoch)
        tensorboard.add_scalar('weight_decay', optimizer.param_groups[0]['weight_decay'])
        tensorboard.add_scalar('weight_effective_bits', weight_effective_bits)
        tensorboard.add_scalar('act_effective_bits', act_effective_bits)
        tensorboard.add_scalar('train/loss', train_loss, epoch)
        tensorboard.add_scalar('train/acc', train_acc, epoch)

        # Run validation test and log stats
        val_loss, val_acc = inference(net, criterion, device, validate_loader)
        tensorboard.add_scalar('val/loss', val_loss, epoch)
        tensorboard.add_scalar('val/acc', val_acc, epoch)

        if best_val_acc is None or val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), net_filename)
        logger.info(f'Val acc = {val_acc}, best_acc = {best_val_acc}')

        scheduler.step()
        tensorboard.flush()

        if epoch > 0 and epoch % 10 == 0:
            if weight_bits:
                weight_effective_bits = max(weight_effective_bits - 1, quantize_weights_bits)
                net.set_quantize_weights(weight_effective_bits)

            if act_bits:
                act_effective_bits = max(act_effective_bits - 1, quantize_activations_bits)
                net.set_quantize_activations(act_effective_bits)

    # If the last model is not the best one, load the best one for testing
    if val_acc != best_val_acc:
        best_state = torch.load(net_filename, map_location=device)
        net.load_state_dict(best_state)
    test_acc = test_net(net, device, test_loader)

    tensorboard.add_scalar('test/acc', test_acc)
    logger.info(f'Network test acc = {test_acc}')

    save_summary({'test_acc': test_acc}, results_dir)


def train_distillation(config) -> None:
    name = config['net']
    dataset = config['dataset']
    dataset_dir = config['dataset_dir']
    batch_size = config['batch_size']
    train_part, val_part = config['train_val_split']
    num_workers = config['num_workers']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = config['results_dir']
    criterion = F.cross_entropy
    lr = config['lr']
    weight_decay = config['weight_decay']
    epoch_num = config['epoch_num']

    stable = config['stable']
    stability_coeff = config['stability_coeff']
    highp_filename = os.path.join(results_dir, name + '_highp.pth')
    lowp_filename = os.path.join(results_dir, name + '_lowp.pth')
    highp_bits = config['highp_bits']
    lowp_bits = config['lowp_bits']
    activation_loss_coeff = config['activation_loss_coeff']

    # Prepare dataset
    train_loader, validate_loader, test_loader, classes = \
        load_dataset(dataset, batch_size, train_part, val_part, num_workers,
                     data_root=os.path.join(dataset_dir, dataset))

    # Prepare networks
    highp_net = make_net(arch=name, classes=classes,
                         stable=stable, stability_coeff=stability_coeff,
                         quantize_weights=lowp_bits, quantize_activations=highp_bits,
                         return_layer_activations=True).to(device)
    lowp_net = make_net(arch=name, classes=classes,
                        stable=stable, stability_coeff=stability_coeff,
                        quantize_weights=lowp_bits, quantize_activations=lowp_bits,
                        return_layer_activations=True).to(device)
    if device == 'cuda':
        highp_net = torch.nn.DataParallel(highp_net)
        lowp_net = torch.nn.DataParallel(lowp_net)
        cudnn.benchmark = True

    # Share weights from highp_net to lowp_net
    highp_params = []
    lowp_params = []
    for key, param in highp_net.named_parameters():
        if '.act' in key and 'alpha' in key:
            highp_params.append(param)
            if key in dict(lowp_net.named_parameters()):
                lowp_params.append(lowp_net.get_parameter(key))
        elif key in dict(lowp_net.named_parameters()):
            lowp_net.get_parameter(key).data = param.data
            lowp_params.append(param)
        else:
            highp_params.append(param)

    wandb.watch(highp_net, log='all', log_freq=len(train_loader.dataset) // batch_size)
    wandb.watch(lowp_net, log='all', log_freq=len(train_loader.dataset) // batch_size)

    optim_params = [{'params': lowp_params},
                    {'params': highp_params}]
    optimizer = SGD(optim_params, lr, 0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
    tensorboard = SummaryWriter(results_dir)

    highp_best_val_acc, lowp_best_val_acc = 0, 0
    for epoch in range(epoch_num):
        logger.info(f'Epoch {epoch + 1}/{epoch_num}')

        highp_correct, lowp_correct = 0, 0
        losses, highp_losses, lowp_losses, act_losses = [], [], [], []

        highp_net.train()
        lowp_net.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            highp_outputs, highp_activations = highp_net(inputs)
            lowp_outputs, lowp_activations = lowp_net(inputs)

            highp_loss = criterion(highp_outputs, targets)
            lowp_loss = criterion(lowp_outputs, targets)
            act_loss = activation_loss_coeff * activation_loss(device, lowp_bits,
                                                               highp_activations, lowp_activations)
            loss = highp_loss + lowp_loss + act_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            highp_losses.append(highp_loss.item())
            lowp_losses.append(lowp_loss.item())
            act_losses.append(act_loss.item())

            _, highp_predicted = highp_outputs.max(1)
            _, lowp_predicted = lowp_outputs.max(1)

            highp_correct += highp_predicted.eq(targets).sum().item()
            lowp_correct += lowp_predicted.eq(targets).sum().item()

        effective_lr = optimizer.param_groups[0]['lr']
        logger.info(f'LR: {effective_lr}')
        tensorboard.add_scalar('lr', effective_lr, epoch)
        tensorboard.add_scalar('weight_decay', optimizer.param_groups[0]['weight_decay'])

        highp_test_acc = highp_correct / len(train_loader.dataset)
        lowp_test_acc = lowp_correct / len(train_loader.dataset)
        logger.info(f'Train acc = highp_net {highp_test_acc} lowp_net {lowp_test_acc}')
        tensorboard.add_scalar('train/acc/highp', highp_test_acc, epoch)
        tensorboard.add_scalar('train/acc/lowp', lowp_test_acc, epoch)

        loss = sum(losses) / (batch_idx + 1)
        highp_loss = sum(highp_losses) / (batch_idx + 1)
        lowp_loss = sum(lowp_losses) / (batch_idx + 1)
        act_loss = sum(act_losses) / (batch_idx + 1)
        tensorboard.add_scalar('train/loss', loss, epoch)
        tensorboard.add_scalar('train/loss/highp', highp_loss, epoch)
        tensorboard.add_scalar('train/loss/lowp', lowp_loss, epoch)
        tensorboard.add_scalar('train/loss/activation', act_loss, epoch)

        highp_val_loss, highp_val_acc = inference(highp_net, criterion, device, validate_loader)
        lowp_val_loss, lowp_val_acc = inference(lowp_net, criterion, device, validate_loader)
        tensorboard.add_scalar('val/loss/highp', highp_val_loss, epoch)
        tensorboard.add_scalar('val/acc/highp', highp_val_acc, epoch)
        tensorboard.add_scalar('val/loss/lowp', lowp_val_loss, epoch)
        tensorboard.add_scalar('val/acc/lowp', lowp_val_acc, epoch)

        if highp_val_acc > highp_best_val_acc:
            highp_best_val_acc = highp_val_acc
            torch.save(highp_net.state_dict(), highp_filename)
        if lowp_val_acc > lowp_best_val_acc:
            lowp_best_val_acc = lowp_val_acc
            torch.save(lowp_net.state_dict(), lowp_filename)

        logger.info(f'Val highp_net acc = {highp_val_acc}, best_acc = {highp_best_val_acc}')
        logger.info(f'Val lowp_net acc = {lowp_val_acc}, best_acc = {lowp_best_val_acc}')

        scheduler.step()
        tensorboard.flush()

    # If the last models are not the best ones, load the best models for testing
    if highp_val_acc != highp_best_val_acc:
        best_state = torch.load(highp_filename, map_location=device)
        highp_net.load_state_dict(best_state)
    if lowp_val_acc != lowp_best_val_acc:
        best_state = torch.load(lowp_filename, map_location=device)
        highp_net.load_state_dict(best_state)

    highp_test_acc = test_net(highp_net, device, test_loader)
    lowp_test_acc = test_net(lowp_net, device, test_loader)
    save_summary({'highp_test_acc': highp_test_acc, 'lowp_test_acc': lowp_test_acc}, results_dir)
    tensorboard.add_scalar('test/acc/highp', highp_test_acc)
    tensorboard.add_scalar('test/acc/lowp', lowp_test_acc)
    logger.info(f'highp test acc = {highp_test_acc}')
    logger.info(f'lowp test acc = {lowp_test_acc}')


def train_post_similarity(config) -> None:
    experiment = load_experiment(config['experiment_path'])
    dataset = experiment['dataset']
    dataset_dir = experiment['dataset_dir']
    lowp_act_bits = experiment['quantize_activations']

    batch_size = config['batch_size']
    epoch_num = config['epoch_num']
    weight_decay = config['weight_decay']
    train_part, val_part = config['train_val_split']
    num_workers = config['num_workers']
    results_dir = config['results_dir']

    target_similarity = config['target_similarity']
    activation_loss_coeff = config['activation_loss_coeff']

    # Prepare dataset
    train_loader, validate_loader, test_loader, classes = \
        load_dataset(dataset, batch_size, train_part, val_part, num_workers,
                     data_root=os.path.join(dataset_dir, dataset))

    # Prepare networks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    highp_net = load_net_from(experiment, classes, device)
    lowp_net = load_net_from(experiment, classes, device)
    highp_net.set_quantize_activations(32)
    if device == 'cuda':
        highp_net = torch.nn.DataParallel(highp_net)
        lowp_net = torch.nn.DataParallel(lowp_net)
        cudnn.benchmark = True

    wandb.watch(highp_net, log='all', log_freq=len(train_loader.dataset) // batch_size)
    wandb.watch(lowp_net, log='all', log_freq=len(train_loader.dataset) // batch_size)

    optim_params = [{'params': lowp_net.parameters()}, ]
    optimizer = SGD(optim_params, 0.01, 0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num)
    tensorboard = SummaryWriter(results_dir)

    for epoch in range(epoch_num):
        logger.info(f'Epoch {epoch + 1}/{epoch_num}')

        highp_correct, lowp_correct = 0, 0
        losses, act_losses = [], []

        highp_net.train()
        lowp_net.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            highp_outputs, highp_activations = highp_net(inputs)
            lowp_outputs, lowp_activations = lowp_net(inputs)

            act_loss = activation_loss(device, lowp_act_bits, highp_activations, lowp_activations)
            loss = activation_loss_coeff * act_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            act_losses.append(act_loss.item())

            _, highp_predicted = highp_outputs.max(1)
            _, lowp_predicted = lowp_outputs.max(1)

            highp_correct += highp_predicted.eq(targets).sum().item()
            lowp_correct += lowp_predicted.eq(targets).sum().item()

        effective_lr = optimizer.param_groups[0]['lr']
        logger.info(f'LR: {effective_lr}')
        tensorboard.add_scalar('lr', effective_lr, epoch)
        tensorboard.add_scalar('weight_decay', optimizer.param_groups[0]['weight_decay'])

        highp_test_acc = highp_correct / len(train_loader.dataset)
        lowp_test_acc = lowp_correct / len(train_loader.dataset)
        logger.info(f'Train acc = highp_net {highp_test_acc} lowp_net {lowp_test_acc}')
        tensorboard.add_scalar('train/acc/highp', highp_test_acc, epoch)
        tensorboard.add_scalar('train/acc/lowp', lowp_test_acc, epoch)

        loss = sum(losses) / (batch_idx + 1)
        act_loss = sum(act_losses) / (batch_idx + 1)
        tensorboard.add_scalar('train/loss', loss, epoch)
        tensorboard.add_scalar('train/loss/activation', act_loss, epoch)

        torch.save(highp_net.state_dict(), os.path.join(results_dir, f'highp_net_{epoch}.pth'))
        torch.save(lowp_net.state_dict(), os.path.join(results_dir, f'lowp_net_{epoch}.pth'))

        scheduler.step()
        tensorboard.flush()

        if np.isclose(act_loss, target_similarity) or act_loss < target_similarity:
            break

    highp_test_acc = test_net(highp_net, device, test_loader)
    lowp_test_acc = test_net(lowp_net, device, test_loader)
    save_summary({'highp_test_acc': highp_test_acc, 'lowp_test_acc': lowp_test_acc}, results_dir)
    tensorboard.add_scalar('test/acc/highp', highp_test_acc)
    tensorboard.add_scalar('test/acc/lowp', lowp_test_acc)
    logger.info(f'highp test acc = {highp_test_acc}')
    logger.info(f'lowp test acc = {lowp_test_acc}')


def report_singles_glob(config) -> None:
    experiment_summaries = {}
    for experiment_path in glob_dirs(config['experiment_path']):
        experiment_mode = load_experiment(experiment_path)
        experiment_label = experiment_mode['experiment_label']

        logger.info(f'Testing model from experiment {experiment_label}: {experiment_path}')

        test_loader, classes = load_test_dataset_for(experiment_mode)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = load_net_from(experiment_mode, classes, device, False)
        test_acc = test_net(net, device, test_loader)
        summary = {'test_acc': test_acc}
        save_summary(summary, experiment_path)

        experiment_summaries[experiment_label] = summary

    logger.info(f'{"Label":<15} {"Test Acc":<15}')
    for label, summary in experiment_summaries.items():
        test_acc = summary['test_acc']
        logger.info(f"{label:<15} {test_acc:<15}")


def report_compare_net_to_fp(config) -> None:
    experiment = load_experiment(config['experiment_path'])

    test_loader, classes = load_test_dataset_for(experiment)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    highp_net = load_net_from(experiment, classes, device, 'all')
    lowp_net = load_net_from(experiment, classes, device, 'all')
    highp_net.set_quantize_activations(32)

    highp_net.eval()
    lowp_net.eval()
    correct = 0
    total_numel = 0
    mse = torch.as_tensor(0.0, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            highp_outputs, highp_activations = highp_net(inputs)
            _, lowp_activations = lowp_net(inputs)

            for ha, la in zip(highp_activations, lowp_activations):
                mse += torch.norm(ha - la) ** 2
                total_numel += ha.numel()

            _, predicted = highp_outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        mse = torch.sqrt(mse / total_numel).item()

        acc = correct / len(test_loader.dataset)

    summary = {'mse': mse,
               'test/acc/highp': acc}
    save_summary(summary, experiment['results_dir'])


def report_graph_activation_diffs(config) -> None:
    results_dir = config['results_dir']

    unstable_experiment = load_experiment(config['unstable_experiment'])
    stable_experiment = load_experiment(config['stable_experiment'])
    experiments = [unstable_experiment, stable_experiment]
    assert len(set(exp['net_type'] for exp in experiments)) == 1
    assert len(set(exp['dataset'] for exp in experiments)) == 1
    assert len(set(exp['quantize_weights'] for exp in experiments)) == 1
    assert len(set(exp['quantize_activations'] for exp in experiments)) == 1

    test_loader, classes = load_test_dataset_for(stable_experiment)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unstable_net = load_net_from(unstable_experiment, classes, device, 'all')
    stable_net = load_net_from(stable_experiment, classes, device, 'all')

    unstable_fp_net = load_net_from(unstable_experiment, classes, device, 'all')
    unstable_fp_net.set_quantize_activations(32)
    stable_fp_net = load_net_from(stable_experiment, classes, device, 'all')
    stable_fp_net.set_quantize_activations(32)

    def mse(a, aq):
        return torch.norm(a - aq) ** 2

    def mses(act, act_q):
        return torch.tensor(list(map(mse, act, act_q)), device=device)

    def numels(tensors):
        return torch.tensor(list(map(torch.numel, tensors)), device=device)

    unstable_net.eval()
    stable_net.eval()
    unstable_mses = torch.as_tensor(0.0, dtype=torch.float64, device=device)
    stable_mses = torch.as_tensor(0.0, dtype=torch.float64, device=device)
    unstable_numels = torch.tensor(0, dtype=torch.int64, device=device)
    stable_numels = torch.tensor(0, dtype=torch.int64, device=device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, unstable_activations = unstable_net(inputs)
            _, unstable_fp_activations = unstable_fp_net(inputs)
            _, stable_activations = stable_net(inputs)
            _, stable_fp_activations = stable_fp_net(inputs)

            unstable_mses = unstable_mses + \
                            mses(unstable_activations, unstable_fp_activations)
            unstable_numels = unstable_numels + numels(unstable_activations)
            stable_mses = stable_mses + \
                          mses(stable_activations, stable_fp_activations)
            stable_numels = stable_numels + numels(stable_activations)

    unstable_mses = torch.sqrt(unstable_mses / unstable_numels).tolist()
    stable_mses = torch.sqrt(stable_mses / stable_numels).tolist()
    save_summary({'unstable_mses': unstable_mses, 'stable_mses': stable_mses}, results_dir)

    plt.plot(range(len(unstable_mses)), unstable_mses, label="unstable resnet56")
    plt.plot(range(len(stable_mses)), stable_mses, label="stable resnet56")
    plt.legend()
    plt.show()

    csv_writer = csv.writer(open(os.path.join(results_dir, 'summary_data.csv'), 'w'))
    csv_writer.writerow(('unstable', 'stable', 'layer'))
    csv_writer.writerows(zip(unstable_mses, stable_mses, range(1, len(unstable_mses) + 1)))


def report_num_parameters(config) -> None:
    def count_net_params(arch, classes, stable, stability_coeff=0.1):
        net = make_net(arch=arch, classes=classes,
                       stable=stable, stability_coeff=stability_coeff,
                       quantize_weights=None, quantize_activations=None)
        return sum(p.numel() for p in net.parameters() if p.requires_grad)

    cifar10 = dataset_classes['cifar10']
    cifar100 = dataset_classes['cifar100']
    params = {'resnet18_unstable_cifar10':
                  count_net_params('resnet18', cifar10, False),
              'resnet56_unstable_cifar10':
                  count_net_params('resnet56', cifar10, False),
              'resnet56_stable_cifar10':
                  count_net_params('resnet56', cifar10, True),
              'stablenet56_unstable_cifar10':
                  count_net_params('stablenet56', cifar10, False),
              'stablenet56_stable_cifar10':
                  count_net_params('stablenet56', cifar10, True),
              'mobilenetv2_unstable_cifar10':
                  count_net_params('mobilenetv2', cifar10, False),
              'mobilenetv2_stable_cifar10':
                  count_net_params('mobilenetv2', cifar10, True),
              'resnet34_unstable_cifar100':
                  count_net_params('resnet34', cifar100, False),
              'resnet34_stable_cifar100':
                  count_net_params('resnet34', cifar100, True),
              'stablenet34_unstable_cifar100':
                  count_net_params('stablenet34', cifar100, False),
              'stablenet34_stable_cifar100':
                  count_net_params('stablenet34', cifar100, True),
              'resnet56_unstable_cifar100':
                  count_net_params('resnet56', cifar100, False),
              'resnet56_stable_cifar100':
                  count_net_params('resnet56', cifar100, True),
              'stablenet56_unstable_cifar100':
                  count_net_params('stablenet56', cifar100, False),
              'stablenet56_stable_cifar100':
                  count_net_params('stablenet56', cifar100, True),
              'mobilenetv2_unstable_cifar100':
                  count_net_params('mobilenetv2', cifar100, False),
              'mobilenetv2_stable_cifar100':
                  count_net_params('mobilenetv2', cifar100, True),
              }

    print(params)


def train_step(net, optimizer, criterion, train_loader, device):
    correct = 0
    total_loss = 0

    net.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        loss = criterion(outputs, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)

        correct += predicted.eq(targets).sum().item()

        loss_progress(batch_idx, loss, train_loader)

    total_loss = total_loss / (batch_idx + 1)
    acc = correct / len(train_loader.dataset)
    return total_loss, acc


def inference(net, criterion, device, data_loader):
    total_loss, correct = 0.0, 0.0

    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            if len(outputs) == 2:
                outputs, _ = outputs
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            loss_progress(batch_idx, loss, data_loader)

    total_loss = total_loss / (batch_idx + 1)
    acc = correct / len(data_loader.dataset)

    return total_loss, acc


def loss_progress(batch_idx, loss, loader):
    if batch_idx % 100 == 0:
        pct = 100. * batch_idx / len(loader)
        logger.debug(f'{pct:.0f}%\tLoss: {loss.item():.6f}')


def init():
    if 'log_file' not in CONFIG:
        CONFIG['log_file'] = os.path.join(CONFIG['results_dir'], 'output.log')

    save_experiment(CONFIG, CONFIG['results_dir'])

    init_wandb()

    init_logger(CONFIG['log_file'])
    logger.info('Running mode:')
    logger.info(pprint.pformat(CONFIG))


class Experiments:
    def single(self, **cmd_line_opts):
        CONFIG['experiment_type'] = 'single'
        update_global_config(cmd_line_opts)
        if 'results_dir' not in cmd_line_opts:
            CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'], CONFIG['net_type'],
                                                 CONFIG['net'], 'single',
                                                 CONFIG['dataset'],
                                                 f'qw{CONFIG["quantize_weights"]}_qa{CONFIG["quantize_activations"]}',
                                                 CONFIG['experiment_label'])

        init()

        train_single(CONFIG)

    def distil(self, **cmd_line_opts):
        CONFIG['experiment_type'] = 'distil'
        update_global_config(cmd_line_opts)

        init()

        train_distillation(CONFIG)

    def post_sim(self, **cmd_line_opts):
        CONFIG['experiment_type'] = 'post_sim'
        update_global_config(cmd_line_opts)

        init()

        train_post_similarity(CONFIG)

    def report_singles(self, **cmd_line_opts):
        CONFIG['experiment_type'] = 'report_singles'
        update_global_config(cmd_line_opts)
        if 'results_dir' not in cmd_line_opts:
            CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'],
                                                 CONFIG['experiment_type'],
                                                 CONFIG['experiment_label'])

        init()

        report_singles_glob(CONFIG)

    def report_cmp_net_to_fp(self, **cmd_line_opts) -> None:
        CONFIG['experiment_type'] = 'report_cmp_net_to_fp'
        update_global_config(cmd_line_opts)
        if 'results_dir' not in cmd_line_opts:
            CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'],
                                                 CONFIG['experiment_type'],
                                                 CONFIG['experiment_label'])

        init()

        report_compare_net_to_fp(CONFIG)

    def report_graph_act_diffs(self, **cmd_line_opts) -> None:
        CONFIG['experiment_type'] = 'report_graph_act_diffs'
        update_global_config(cmd_line_opts)
        if 'results_dir' not in cmd_line_opts:
            CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'],
                                                 CONFIG['experiment_type'],
                                                 CONFIG['experiment_label'])

        init()

        report_graph_activation_diffs(CONFIG)

    def report_num_parameters(self, **cmd_line_opts) -> None:
        CONFIG['experiment_type'] = 'report_num_parameters'
        update_global_config(cmd_line_opts)
        if 'results_dir' not in cmd_line_opts:
            CONFIG['results_dir'] = os.path.join(CONFIG['checkpoint_dir'],
                                                 CONFIG['experiment_type'],
                                                 CONFIG['experiment_label'])

        init()

        report_num_parameters(CONFIG)

    def replay(self, experiment: str, **override_cmd_line_opts):
        if os.path.isfile(experiment) and experiment.endswith('.json'):
            experiment = os.path.dirname(experiment)

        config = load_experiment(experiment)

        CONFIG.clear()
        CONFIG.update(config)
        CONFIG['experiment_replay'] = experiment
        del CONFIG['experiment_label']
        del CONFIG['results_dir']
        del CONFIG['log_file']

        experiment_type = CONFIG['experiment_type']
        if hasattr(self, experiment_type) and callable(getattr(self, experiment_type)):
            getattr(self, experiment_type)(**override_cmd_line_opts)
        else:
            raise ValueError(f'Experiment type {experiment_type} not supported')


if __name__ == '__main__':
    fire.Fire(Experiments)
