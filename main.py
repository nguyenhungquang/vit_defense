import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
import math
import torchvision
import glob
#from matplotlib import pyplot as plt
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
from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision import datasets as dset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader, Subset

from attack.simba import *
from attack.square_attack import *
from attack.bandit import *
from attack.nes import NES
from attack.nes_adapt import NES_Adaptive
from attack.signhunt import SignHunt
from attack.decision import *
from models.robust_vit import VisionTransformer
from models.utils import create_model
from models.wrapper import ModelWrapper

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




# def main(args):



if __name__ == '__main__':
    attack_list = ['square_linf', 'square_l2', 'simba_dct', 'nes_l2', 'nes_linf', 'bandit_l2', 'bandit_linf', 'signhunt_l2', 'hsja_l2', 'nes_adapt_linf', 'nes_adapt_l2']
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='Model name.')
    parser.add_argument('--attack', type=str, default='square_linf', choices=attack_list, help='Attack.')
    parser.add_argument('--exp_folder', type=str, default='exps_balance_mean', help='Experiment folder to store all output.')
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
    parser.add_argument('--layer_index', nargs='*', default=-1)
    parser.add_argument('--noise_list', nargs='*', default=None)
    parser.add_argument('--stop_criterion', type=str, default='fast_exp', choices=['single', 'without_defense', 'fast_exp', 'exp', 'none'])
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--num_adapt', type=int, default=1)
    # args = parser.parse_args(
    #     '--attack=square_linf --model=pt_vit_base_patch16_224_cifar10_finetuned --n_ex=1000 --eps=12.75 --p=0.05 --n_iter=10000'.split(' '))
    args = parser.parse_args()
        # '--attack=square_linf --model=resnet50_cifar10 --n_ex=1000 --eps=12.75 --p=0.05 --n_iter=10000'.split(' '))
    try:
        if not isinstance(args.layer_index, int):
            if len(args.layer_index) == 1:
                args.layer_index = int(args.layer_index[0])
            else:
                args.layer_index = [int(_) for _ in args.layer_index]
        print(args.layer_index)
        args.loss = 'margin_loss' if not args.targeted else 'cross_entropy'

        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device(f'cuda:{args.gpu}')
        dataset = args.dataset#'mnist' if 'mnist' in args.model else 'cifar10' if 'cifar10' in args.model else 'imagenet'
        timestamp = str(datetime.now())[:-7]
        
        # args.eps = args.eps / 255.0 if dataset == 'imagenet' else args.eps  # for mnist and cifar10 we leave as it is
        batch_size = 50 #data.bs_dict[dataset]
        n_cls = 1000 if dataset == 'imagenet' else 10
        gpu_memory = 0.5 if dataset == 'mnist' and args.n_ex > 1000 else 0.15 if dataset == 'mnist' else 0.99

        # log_path = '{}/{}.log'.format(args.exp_folder, hps_str)
        # metrics_path = '{}/{}.metrics'.format(args.exp_folder, hps_str)

        # log = utils.Logger(log_path)
        # log.print('All hps: {}'.format(hps_str))
        # print('All hps: {}'.format(hps_str))
        cfg = timm.create_model(args.model).default_cfg
        scale_size = int(math.floor(cfg['input_size'][-2]))
        if cfg['interpolation'] == 'bilinear':
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR 
        elif cfg['interpolation'] == 'bicubic':
            interpolation = torchvision.transforms.InterpolationMode.BICUBIC
        transform = Compose([
            # Resize(224),
            Resize(scale_size, interpolation=interpolation),
            CenterCrop(size=(224, 224)),
            ToTensor()#,
            # Normalize(timm.data.constants.IMAGENET_DEFAULT_MEAN, timm.data.constants.IMAGENET_DEFAULT_STD)   
        ])
        if 'square' in args.attack or 'hsja' in args.attack:
            use_numpy = True
        else:
            use_numpy = False
        
        # dataset_name = 'cifar10'
        if os.path.exists(f'cache/{args.dataset}.pth'):
            x_test, y_test = torch.load(f'cache/{args.dataset}.pth')
            # breakpoint()
            y_test = y_test.to(int)
        else:
            if args.dataset == 'cifar10':
                dataset = dset.CIFAR10('~/quang.nh/data', train=False, download=True, transform=transform)  
            elif args.dataset == 'imagenet':
                dataset = dset.ImageFolder('data/imagenet/val', transform=transform)
            # loader = DataLoader(dataset, batch_size=args.n_ex, shuffle=False, num_workers=8)
            # x_test, y_test = next(iter(loader))
            def get_val_data(dataset, n_cls, n_ex):
                assert n_ex % n_cls == 0
                n_samples_each = n_ex // n_cls
                loader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=False, pin_memory=True)
                iter_loader = iter(loader)
                img_size = dataset[0][0].shape
                # x = torch.zeros(n_cls, n_samples_each, *img_size)
                x = [torch.zeros(0, *img_size) for _ in range(n_cls)]
                # y = torch.ones(n_cls, n_samples_each) * -1
                y = [torch.zeros(0) for _ in range(n_cls)]
                while sum([len(_) != n_samples_each for _ in y]) > 0:
                    # print(sum([len(_) != n_samples_each for _ in y]))
                    img, labels = next(iter_loader)
                    for l in torch.unique(labels):
                        # img[l, :c] 
                        curr_n_ex = y[l].shape[0]
                        if curr_n_ex < n_samples_each:
                            x[l] = torch.cat([x[l], img[labels == l][:n_samples_each - curr_n_ex]])
                            y[l] = torch.cat([y[l], labels[labels == l][:n_samples_each - curr_n_ex]])
                        # x[l].append(img[labels == l])
                        # y[l].append(labels[labels == l])
                x = torch.cat(x)
                y = torch.cat(y).to(int)
                return x, y
            x_test, y_test = get_val_data(dataset, n_cls, args.n_ex)
            torch.save((x_test, y_test), f'cache/{args.dataset}.pth')

        
        # print('Done load data')
        
        # print(x_test.min(), x_test.max())
        # # print('Convert x')
        if use_numpy:
            x_test = x_test.numpy()
            y_test = y_test.numpy()

        config_pth = f'configs/{args.attack}.yaml'
        with open(config_pth, 'r') as fr:
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
            # noise_list = [2/255, 3/255, 4/255, 5/255][8/255, 7/255, 6/255]
            # noise_list = [8/255, 7/255, 6/255]
            noise_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        elif args.def_position == 'hidden_feature':
            # assert 'resnet' in args.model
            noise_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.15, 0.2, 0.25]
        elif args.def_position == 'logits':
            noise_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]
        elif args.def_position == 'last_cls':
            noise_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        elif args.def_position == 'pre_att_all':
            # noise_list = [0.2, 0.1, 0.075, 0.05, 0.01, 0.005]
            noise_list = [0.03, 0.05]
        elif args.def_position == 'pre_att_cls':
            # noise_list = [0.4, 0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.01, 0.005]
            noise_list = [0.05, 0.1, 0.15]
        elif args.def_position == 'post_att_all':
            noise_list = [0.075, 0.05, 0.01, 0.005]
        elif args.def_position == 'post_att_cls':
            noise_list = [0.3, 0.2, 0.15, 0.1, 0.075, 0.05, 0.01, 0.005]
        if args.noise_list is not None:
            noise_list = [float(_) for _ in args.noise_list]
        for noise in noise_list:
        # for noise in noise_list:
            print('Start running for noise scale', noise)
            hps_str = 'model={} defense={} n_ex={} eps={} p={} n_iter={} noise_scale={}'.format(
            args.model, args.defense, args.n_ex, args.eps, args.p, args.n_iter, noise)
            method = args.def_position#'last_cls'
            base_dir = '{}/{}/{}/{}/{}/'.format(args.model, args.exp_folder, args.attack + '_' + str(args.eps) + '_' + str(config), args.dataset, method + f'_layer_{args.layer_index}')
            # base_dir = 'debug/'
            log_dir = base_dir + 'log/'
            os.makedirs(log_dir, exist_ok=True)
            log_path = log_dir + f'{hps_str}.log'
            metrics_path = base_dir + 'metrics/'
            os.makedirs(metrics_path, exist_ok=True)
            metrics_path += hps_str + '.metrics'

            log = utils.Logger(log_path)
            log.print(str(args.__dict__))
            log.print(str(config))
            # mixed
            if args.def_position == 'mix_cls_first':
                args.def_position = ['pre_att_cls'] * 7 + ['pre_att_all'] * 5
            elif args.def_position == 'mix_all_first':
                noise = [0.05] * 7 + [0.1] * 5
                args.def_position = ['pre_att_all'] * 7 + ['pre_att_cls'] * 5
            
            # if not 'vit' in args.model:
            #     base_model = timm.create_model(args.model, num_classes=None)
                
            # else:
            model = create_model(args.model, args.dataset, n_cls, noise, args.defense, args.def_position, device=device)
            print('done load model')
            logits_clean = model.predict(x_test, not use_numpy)
            corr_classified = (logits_clean.argmax(1) == y_test)
            acc = corr_classified.float().mean() if torch.is_tensor(corr_classified) else corr_classified.mean()
            # important to check that the model was restored correctly and the clean accuracy is high
            log.print('Clean accuracy: {:.2%}'.format(acc))
            # if acc < 0.75:
            #     continue 

            if args.attack == 'simba_dct':
                attacker = SimBA(model, log, args.eps, 224, **config)
                # _, prob, succ, queries, l2, linf = attacker.attack(x_test[corr_classified], y_test[corr_classified], args.n_iter)
                _, prob, succ, queries, l2, linf = attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'nes' in args.attack and not 'adapt' in args.attack:
                # lp = args.attack.split('_')[1]
                attacker = NES(model, log, args.eps, **config)
                # y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
                # y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'nes_adapt' in args.attack:
                # lp = args.attack.split('_')[1]
                attacker = NES_Adaptive(model, log, args.eps, **config)
                # y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
                # y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'bandit' in args.attack:
                # lp = args.attack.split('_')[1]
                attacker = Bandit(model, log, args.eps, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'signhunt' in args.attack:
                # lp = args.attack.split('_')[1]
                attacker = SignHunt(model, log, **config)
                attacker.attack(x_test, y_test, args.n_iter, args.stop_criterion)
            elif 'square' in args.attack:
                square_attack = square_attack_linf if args.attack == 'square_linf' else square_attack_l2
                # y_target = utils.random_classes_except_current(y_test, n_cls) if args.targeted else y_test
                # y_target_onehot = utils.dense_to_onehot(y_target, n_cls=n_cls)
                # Note: we count the queries only across correctly classified images
                n_queries, x_adv = square_attack(model, x_test, y_test, corr_classified, args.eps, args.n_iter,
                                                args.p, metrics_path, args.targeted, args.loss, log, args.stop_criterion, adaptive=args.adaptive, M=args.num_adapt)
                print(f'Noise :{noise}, n queries: {n_queries}')
            elif 'hsja' in args.attack:
                attacker = HSJAttack(epsilon=args.eps, max_queries=args.n_iter, lb=0, ub=1, batch_size=args.n_ex, use_numpy=use_numpy, **config)
                attacker.run(x_test, y_test, model, args.targeted)
    except KeyboardInterrupt:
        pass
