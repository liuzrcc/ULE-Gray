import copy
import os
import collections
import numpy as np
import torch
import util
import random
import mlconfig
import pandas
from util import onehot, rand_bbox
from torch.utils.data.dataset import Dataset
from functools import partial
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fast_autoaugment.FastAutoAugment.archive import fa_reduced_cifar10
from fast_autoaugment.FastAutoAugment.augmentations import apply_augment
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Datasets
transform_options = {
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR10GRAY": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.Grayscale(3),
                            transforms.ToTensor()],
        "test_transform": [transforms.Grayscale(3),
                           transforms.ToTensor()]},
}

transform_options['PoisonCIFAR10GRAY'] = transform_options['CIFAR10GRAY']
transform_options['PoisonCIFAR10'] = transform_options['CIFAR10']


@mlconfig.register
class DatasetGenerator():
    def __init__(self, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='CIFAR10', seed=0,
                 test_data_path='../datasets/', test_data_type='CIFAR10', fa=False,
                 no_train_augments=False, poison_rate=1.0, perturb_type='classwise',
                 perturb_tensor_filepath=None, patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 use_cutout=None, use_cutmix=False, use_mixup=False):

        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        if no_train_augments:
            train_transform = test_transform

        elif use_cutout is not None:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        # Training Datasets
        if train_data_type == 'CIFAR10':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                             download=True, transform=train_transform)
        elif train_data_type == 'CIFAR10GRAY':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                             download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        elif train_data_type == 'PoisonCIFAR10GRAY':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx)
        else:
            raise('Training Dataset type %s not implemented' % train_data_type)

        # Test Datset
        if test_data_type == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                            download=True, transform=test_transform)
        elif test_data_type == 'CIFAR10GRAY':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                            download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR10':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_classwise_idx=poison_classwise_idx)

        elif test_data_type == 'PoisonCIFAR10GRAY':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_classwise_idx=poison_classwise_idx)

        else:
            raise('Test Dataset type %s not implemented' % test_data_type)

        if use_cutmix:
            train_dataset = CutMix(dataset=train_dataset, num_class=num_of_classes)
        elif use_mixup:
            train_dataset = MixUp(dataset=train_dataset, num_class=num_of_classes)

        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

    def _split_validation_set(self, train_portion, train_shuffle=True, train_drop_last=True):
        np.random.seed(self.seed)
        train_subset = copy.deepcopy(self.datasets['train_dataset'])
        valid_subset = copy.deepcopy(self.datasets['train_dataset'])

        if self.train_data_type == 'ImageNet' or self.train_data_type == 'ImageNetMini' or self.train_data_type == 'TinyImageNet' or self.train_data_type == 'PoisonImageNetMini':
            data, targets = list(zip(*self.datasets['train_dataset'].samples))
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.samples = list(zip(train_D, train_L))
            valid_subset.samples = list(zip(valid_D, valid_L))
        else:
            datasplit = train_test_split(self.datasets['train_dataset'].data,
                                         self.datasets['train_dataset'].targets,
                                         test_size=1-train_portion, train_size=train_portion,
                                         shuffle=True, stratify=self.datasets['train_dataset'].targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.targets = train_L
            valid_subset.targets = valid_L

        self.datasets['train_subset'] = train_subset
        self.datasets['valid_subset'] = valid_subset
        print(self.datasets)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        data_loaders['train_subset'] = DataLoader(dataset=self.datasets['train_subset'],
                                                  batch_size=self.train_batch_size,
                                                  shuffle=train_shuffle, pin_memory=True,
                                                  drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['valid_subset'] = DataLoader(dataset=self.datasets['valid_subset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)
        return data_loaders


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise('Invalid patch location')

    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[x1: x2, y1: y2, :] = noise
    return mask


class PoisonCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None):
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        # Check Shape
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        print(target_dim)
        print(len(self))
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise('Poison Perturb Tensor size not match for classwise')

        # Random Select Poison Targets
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        if poison_classwise:
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx
            self.poison_samples_idx = []
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    self.poison_samples_idx.append(i)
        else:
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class MixUp(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)
