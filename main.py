import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import glob
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import pickle
import cv2
import json
import yaml

from argparse import Namespace
from tqdm.auto import tqdm
import timm
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets as dset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader, Subset

from attack.simba import *
from attack.square_attack import *
from models.robust_vit import VisionTransformer

import logging
logging.getLogger().setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

model_list = ['resnet50', 'vit_base_patch16_224-224']

import math

import argparse
import time
import numpy as np
import data
import models
import os
import utils
from datetime import datetime
np.set_printoptions(precision=5, suppress=True)


class ModelWrapper:
    def __init__(self, model, num_classes=10, def_position=None, device='cpu'):
        self.model = model
        self.num_classes = num_classes
        self.model.to(device)
        self.batch_size = 64
        self.device = device
        # self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
        # self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        self.mean = np.reshape([0.5, 0.5, 0.5], [1, 3, 1, 1])
        self.std = np.reshape([0.5, 0.5, 0.5], [1, 3, 1, 1])
        self.mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        self.std_torch = torch.tensor(self.std, device=device, dtype=torch.float32)
        self.def_position = def_position
        self.model.eval()

    def __call__(self, x):
        if self.def_position == 'input_noise':
            x = x + np.random.normal(scale=2 / 255, size=x.shape).astype(np.float32)
        x = x.to(self.device)
        x = (x - self.mean_torch) / self.std_torch
        return self.model(x).cpu()

    def predict(self, x, return_tensor=False, defense=True):
        self.model.set_defense(defense)
        if self.def_position == 'input_noise' and defense:
            # max_norm = np.random.exponential(scale=1.0, size=None)
            # pert = np.random.uniform(-max_norm, max_norm, size=x.shape)
            # x = x + pert
            x = x + np.random.normal(scale=self.model.noise_sigma, size=x.shape)
        x = (x - self.mean) / self.std
        x = x.float() if torch.is_tensor(x) else x.astype(np.float32)
        if self.def_position == 'feature':
            def forward_new(self, x):
                x = self.forward_features(x)
                x = x + self.noise_sigma * torch.randn_like(x)
                x = self.forward_head(x)
                return x
            import types
            self.model.forward = types.MethodType(forward_new, self.model)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        # loader = DataLoader(x, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=False)
        with torch.no_grad():  # otherwise consumes too much memory and leads to a slowdown
            for i in range(n_batches):
                x_batch = x[i*self.batch_size:(i+1)*self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch).to(self.device)
            # for x_batch in loader:
            #     x_batch_torch = x_batch.to(self.device)
                
                logits = self.model(x_batch_torch)[:, :self.num_classes].cpu()
                if not return_tensor:
                    logits = logits.numpy()
                logits_list.append(logits)
        self.model.set_defense(True)
        if return_tensor:
            logits = torch.cat(logits_list)
        else:
            logits = np.vstack(logits_list)
        return logits

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        """ Implements the margin loss (difference between the correct and 2nd best class). """
        if loss_type == 'margin_loss':
            preds_correct_class = (logits * y).sum(1, keepdims=True)
            diff = preds_correct_class - logits  # difference between the correct class and all other classes
            diff[y] = np.inf  # to exclude zeros coming from f_correct - f_correct
            margin = diff.min(1, keepdims=True)
            loss = margin * -1 if targeted else margin
        elif loss_type == 'cross_entropy':
            probs = utils.softmax(logits)
            loss = -np.log(probs[y])
            loss = loss * -1 if not targeted else loss
        else:
            raise ValueError('Wrong loss.')
        return loss.flatten()

# def main(args):



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', choices=model_list, help='Model name.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=['square_linf', 'square_l2', 'simba_dct'], help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='exps_new', help='Experiment folder to store all output.')
    parser.add_argument('--gpu', type=str, default='3', help='GPU number. Multiple GPUs are possible for PT models.')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--p', type=float, default=0.05,
                        help='Probability of changing a coordinate. Note: check the paper for the best values. '
                            'Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=0.05, help='Radius of the Lp ball.')
    parser.add_argument('--n_iter', type=int, default=10000)
    parser.add_argument('--targeted', action='store_true', help='Targeted or untargeted attack.')
    parser.add_argument('--defense', type=str, default='identical')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--def_position', type=str, default='baseline')
    parser.add_argument('--noise_list', nargs='*', default=None)
    # args = parser.parse_args(
    #     '--attack=square_linf --model=pt_vit_base_patch16_224_cifar10_finetuned --n_ex=1000 --eps=12.75 --p=0.05 --n_iter=10000'.split(' '))
    args = parser.parse_args()
        # '--attack=square_linf --model=resnet50_cifar10 --n_ex=1000 --eps=12.75 --p=0.05 --n_iter=10000'.split(' '))
    try:
        print(args.noise_list)
        args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'

        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device(f'cuda:{args.gpu}')
        dataset = args.dataset#'mnist' if 'mnist' in args.model else 'cifar10' if 'cifar10' in args.model else 'imagenet'
        timestamp = str(datetime.now())[:-7]
        
        # args.eps = args.eps / 255.0 if dataset == 'imagenet' else args.eps  # for mnist and cifar10 we leave as it is
        batch_size = 50 #data.bs_dict[dataset]
        model_type = 'pt' if 'pt_' in args.model else 'tf'
        n_cls = 1000 if dataset == 'imagenet' else 10
        gpu_memory = 0.5 if dataset == 'mnist' and args.n_ex > 1000 else 0.15 if dataset == 'mnist' else 0.99

        # log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
        # metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)

        # log = utils.Logger(log_path)
        # log.print('All hps: {}'.format(hps_str))
        # print('All hps: {}'.format(hps_str))

        transform = Compose([
            Resize(224),
            CenterCrop(size=(224, 224)),
            ToTensor()#,
            # Normalize(timm.data.constants.IMAGENET_DEFAULT_MEAN, timm.data.constants.IMAGENET_DEFAULT_STD)   
        ])
        if 'square' in args.attack:
            use_numpy = True
        else:
            use_numpy = False
        
        # dataset_name = 'cifar10'
        if args.dataset == 'cifar10':
            dataset = dset.CIFAR10('~/quang.nh/data', train=False, download=True, transform=transform)
        elif args.dataset == 'imagenet':
            dataset = dset.ImageFolder('data/imagenet/val', transform=transform)

        loader = DataLoader(dataset, batch_size=args.n_ex, shuffle=False, num_workers=8)
        x_test, y_test = next(iter(loader))
        # print('Done load data')
        
        # print(x_test.min(), x_test.max())
        # # print('Convert x')
        if use_numpy:
            x_test = x_test.numpy()
            y_test = y_test.numpy()

        with open(f'configs/{args.attack}.yaml', 'r') as fr:
            config = yaml.safe_load(fr)

        # x_test = []
        # y_test = []
        # loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)
        # for batch_x, batch_y in tqdm(loader):
        #     x_test.append(batch_x.numpy())
        #     y_test.append(batch_y.numpy())
        # x_test = np.concatenate(x_test)
        # y_test = np.concatenate(y_test)

        dataset_name = args.dataset
        # noise_list = [0] if args.def_position == 'baseline' else [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01]#[1, 1.5, 2, 2.5, 3, 5]
        if args.def_position == 'baseline':
            noise_list = [0]
        elif args.def_position == 'input_noise':
            noise_list = [2/255, 3/255, 4/255, 5/255]
        elif args.def_position == 'logits':
            noise_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
        elif args.def_position == 'last_cls':
            noise_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        elif args.def_position == 'pre_att_all':
            noise_list = [0.1, 0.075, 0.05, 0.01]
        else:
            noise_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01]
        if args.noise_list is not None:
            noise_list = [float(_) for _ in args.noise_list]
        for noise in noise_list:
        # for noise in noise_list:
            print('Start running for noise scale', noise)
            hps_str = 'model={} defense={} n_ex={} eps={} p={} n_iter={} noise_scale={}'.format(
            args.model, args.defense, args.n_ex, args.eps, args.p, args.n_iter, noise)
            method = args.def_position#'last_cls'
            base_dir = '{}/{}/{}/{}/'.format(args.exp_folder, args.attack + '_' + str(args.eps), args.dataset, method)
            # base_dir = 'debug/'
            log_dir = base_dir + 'log/'
            os.makedirs(log_dir, exist_ok=True)
            log_path = log_dir + f'{hps_str}.log'
            metrics_path = base_dir + 'metrics/'
            os.makedirs(metrics_path, exist_ok=True)
            metrics_path += hps_str + '.metrics'

            log = utils.Logger(log_path)
            log.print('All hps: {}'.format(hps_str))
            print('All hps: {}'.format(hps_str))
            
            if not 'vit' in args.model:
                base_model = timm.create_model(args.model, num_classes=None)
                
            else:
                base_model = VisionTransformer(weight_init='skip', num_classes=n_cls, defense_cls=args.defense, noise_sigma=noise, def_position=args.def_position)
            if args.dataset == 'cifar10':
                base_model.load_state_dict(torch.load(f'pretrain/{args.model}_{dataset_name}.pth.tar', map_location='cpu')['state_dict'])
            elif args.dataset == 'imagenet':
                base_model.load_state_dict(timm.create_model(args.model, pretrained=True).state_dict())
            base_model.noise_sigma = noise
            model = ModelWrapper(base_model, num_classes=n_cls, device=device, def_position=args.def_position)
            print('done load model')
            logits_clean = model.predict(x_test, not use_numpy)
            corr_classified = (logits_clean.argmax(1) == y_test)
            acc = corr_classified.float().mean() if torch.is_tensor(corr_classified) else corr_classified.mean()
            # important to check that the model was restored correctly and the clean accuracy is high
            log.print('Clean accuracy: {:.2%}'.format(acc))
            if acc < 0.9:
                continue 

            if args.attack == 'simba_dct':
                attacker = SimBA(model, log, args.eps, 224, **config)
                _, prob, succ, queries, l2, linf = attacker.attack(x_test[corr_classified], y_test[corr_classified], args.n_iter)
            else:
                square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
                y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
                y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
                # Note: we count the queries only across correctly classified images
                n_queries, x_adv = square_attack(model, x_test, y_target_onehot, corr_classified, args.eps, args.n_iter,
                                                args.p, metrics_path, args.targeted, args.loss, log)
                print(f'Noise :{noise}, n queries: {n_queries}')
    except KeyboardInterrupt:
        pass
