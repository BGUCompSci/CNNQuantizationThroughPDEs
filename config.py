import torch

MODEL = {
    'net': 'resnet56',
    'net_type': 'stable'
}

DATA_SET = {
    'dataset': 'cifar10',  # cifar10 | cifar100 | imagenet
    'num_workers': 2,
}

TRAIN = {
    'loss': 'crossentropy',
    'batch_size': 128,
    'epoch_num': 200,
    'lr': 1e-1,
    'weight_decay': 1e-4,
    'train_val_split': (0.9, 0.1),
}

QUANTIZATION = {
    'quantize_activations': False,
    'quantize_weights': False,
}

STABILITY = {
    'stable': True,
    'stability_coeff': 0.1
}

DISTILLATION = {
    'highp_bits': 32,
    'lowp_bits': 8,
    'activation_loss_coeff': 1e-5,
}

WANDB = {
    # Add your WandB account and project name here to upload data to WandB
    'wandb_entity': None
    'wandb_project_name': None,
}

CONFIG = {}
CONFIG.update(MODEL)
CONFIG.update(DATA_SET)
CONFIG.update(TRAIN)
CONFIG.update(QUANTIZATION)
CONFIG.update(STABILITY)
CONFIG.update(DISTILLATION)
CONFIG.update(WANDB)

if torch.cuda.is_available():
    print(
        f'''Running on device {torch.cuda.current_device()}
        name {torch.cuda.get_device_name(device=torch.cuda.current_device())}'''
    )
