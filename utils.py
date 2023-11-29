import os
import torch
import random
import numpy as np
import json
import timm
import torchvision.transforms as transforms
from torchvision import datasets

NUM_CLASSES_DICT = {'imagenet': 1000, 'cifar10': 10, 'cifar100': 100}

def set_device(device):
    os.environ['CUDA_VISIBLE_DEVICES'] = device


def set_random_seed(seed=1234):
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def generate_saving_path(configs):
    # generate the result saving path from configs
    json.dumps(configs, indent=4)

    # remove uninformative configs to simplify the saving path
    for k in ["device", "model_dir", "data_dir", "batch_size", "lr", "epsilon"]:
        configs.pop(k)

    if configs["attack"] == 'DorPatch':
        sub_path = []
        for k in ["num_patch", "patch_budget"]:
            sub_path.append("%s=%s" % (k, configs[k]))
            configs.pop(k)
        subdir = '_'.join(sub_path)
    print(subdir)

    save_path = "_".join(["%s=%s" % (k, v) for k, v in configs.items()])
    save_path = os.path.join("results", save_path, subdir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return save_path


def get_model(dataset_name, model_name, model_dir='pretrained_models'):
    '''
    reference: https://github.com/inspire-group/PatchCleanser/blob/main/utils/setup.py
    '''
    timm_models = ['resnetv2_50x1_bit_distilled',
                   'vit_base_patch16_224', 'resmlp_24_distilled_224']
    model = None
    # load timm model
    for tm in timm_models:
        if model_name in tm:
            model = timm.create_model(tm)
            model.reset_classifier(num_classes=NUM_CLASSES_DICT[dataset_name])
            checkpoint_name = tm + '_cutout2_128_{}.pth'.format(dataset_name)
            checkpoint = torch.load(os.path.join(
                model_dir, dataset_name, checkpoint_name))
            model.load_state_dict(checkpoint['state_dict'])
    return model


def get_normalize(dataset_name, model_name):
    return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])


class NormModel(torch.nn.Module):
    def __init__(self, model, normalize):
        super(NormModel, self).__init__()
        self.model = model
        self.normalize = normalize

    def forward(self, x):
        return self.model(self.normalize(x))


def get_dataset(dataset_name, data_dir='/home/data', train=False, batch_size=128, shuffle=True):
    dataset_dict = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'imagenet': datasets.ImageNet,
    }
    args_dict = {
        'cifar10': {'train': train, 'download': True},
        'cifar100': {'train': train, 'download': True},
        'imagenet': {'split': 'train' if train else 'val'},
    }
    size = 224
    dataset = dataset_dict[dataset_name](
        root=os.path.join(data_dir, dataset_name),
        transform=transforms.Compose([
            transforms.Resize(int(size/0.875)),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor()]),
        **args_dict[dataset_name])
    print('Dataset has {} instances'.format(len(dataset)))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)


def clip(mask, pattern, x, eps):
    # clip the perturbation to make sure the patch is within l2 norm
    delta_x = mask * (pattern -  x)
    l2_norm = torch.norm(delta_x, p=2, dim=(1, 2, 3)).detach()
    down_scale = (eps/l2_norm).clip(max=1.).view(-1, 1, 1, 1)
    return delta_x * down_scale

def convert_float_list_to_str(l):
    return ', '.join(["%.2f" % i for i in l])