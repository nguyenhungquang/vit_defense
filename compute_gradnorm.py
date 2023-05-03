import os
import torch
import torchvision.datasets as dset
from torchvision import transforms
from torch.utils.data import DataLoader
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
from torchvision import datasets as dset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, CenterCrop
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from functools import partial
import types

dataset_name = 'cifar10'
device = 'cuda:3'
# model_name = 'vgg19_bn'
model_name = 'resnet50'
os.makedirs('gradnorm', exist_ok=True)

def get_loader(model_name):
    if 'torchvision' not in model_name:
        cfg = timm.create_model(model_name).default_cfg
        scale_size = int(math.floor(cfg['input_size'][-2] / cfg['crop_pct']))
        if cfg['interpolation'] == 'bilinear':
            interpolation = torchvision.transforms.InterpolationMode.BILINEAR 
        elif cfg['interpolation'] == 'bicubic':
            interpolation = torchvision.transforms.InterpolationMode.BICUBIC
    else:
        scale_size = 224
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    transform = Compose([
        # Resize(224),
        Resize(scale_size, interpolation=interpolation),
        CenterCrop(size=(224, 224)),
        ToTensor()#,
        # Normalize(timm.data.constants.IMAGENET_DEFAULT_MEAN, timm.data.constants.IMAGENET_DEFAULT_STD)   
    ])
    if dataset_name == 'imagenet':
        dataset = dset.ImageFolder('data/imagenet/val', transform=transform)
    elif dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root='~/quang.nh/data', train=False, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
    return loader

def margin_loss(logit, label):
    _, argsort = logit.sort(dim=1, descending=True)
    gt_is_max = argsort[:, 0].eq(label)
    second_max_index = gt_is_max.long() * argsort[:, 1] + (~gt_is_max).long() * argsort[:, 0]
    gt_logit = logit[torch.arange(logit.shape[0]), label]
    second_max_logit = logit[torch.arange(logit.shape[0]), second_max_index]
    return second_max_logit - gt_logit

loader = get_loader(model_name)
base_model = timm.create_model(model_name, num_classes=None, pretrained=True)
if dataset_name == 'cifar10':
    model_path = model_name.replace('.', '_')
    base_model.load_state_dict(torch.load(f'pretrain/{model_path}_{dataset_name}.pth.tar', map_location='cpu')['state_dict'])
mean = base_model.pretrained_cfg['mean']
std = base_model.pretrained_cfg['std']
mean = torch.tensor(np.reshape(mean, [1, 3, 1, 1]), device=device)
std = torch.tensor(np.reshape(std, [1, 3, 1, 1]), device=device)

backward = []

if model_name == 'deit_base_patch16_224.fb_in1k':
    def new_blocks_forward(self, x):
        for i, m in enumerate(self):
            # if m.__class__.__name__ == 'Conv2d':
                
                # print(x.flatten(1).norm(dim=1).shape)
            x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
                # x.register_hook(lambda grad:print(grad.shape))
            x = m(x)
        return x
    base_model.blocks.forward = types.MethodType(new_blocks_forward, base_model.blocks)
elif model_name == 'resmlp_12_224.fb_in1k':
    def new_blocks_forward(self, x):
        for i, m in enumerate(self):
            x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
            x = m(x)
        return x
    base_model.blocks.forward = types.MethodType(new_blocks_forward, base_model.blocks)
elif model_name == 'resnet50':
    def new_forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
        # else:
        x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
        x = self.layer1(x)
        x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
        x = self.layer2(x)
        x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
        x = self.layer3(x)
        x .register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
        x = self.layer4(x)
        return x
    
    base_model.forward_features = types.MethodType(new_forward_features, base_model)
elif model_name == 'vgg19_bn':
    def new_forward_features(self, x):
        for i, m in enumerate(self.features):
            if m.__class__.__name__ == 'Conv2d':
                
                # print(x.flatten(1).norm(dim=1).shape)
                x.register_hook(lambda grad: backward.append(grad.flatten(1).norm(dim=1)))
                # x.register_hook(lambda grad:print(grad.shape))
            x = m(x)
        return x
    
    base_model.forward_features = types.MethodType(new_forward_features, base_model)

from tqdm import tqdm
from torch.autograd import Variable
base_model = base_model.to(device).eval()
base_feat_norm = []
base_x_norm = []
base_backward_values = []
base_logit_norm = []
for img, label in tqdm(loader):
    img = img.to(device)
    label = label.to(device)
    backward = []
    img = (img - mean) / std
    img = Variable(img.float(), requires_grad=True)

    feat = base_model.forward_features(img)
    logit = base_model.forward_head(feat)
    # logit.register_hook(lambda grad: base_logit_norm.append(grad.flatten(1).norm(dim=1)))
    loss = margin_loss(logit, label)
    loss.sum().backward(retain_graph=True)
    base_backward_values.append(torch.stack(backward))
    base_x_norm.append(img.grad.flatten(1).norm(dim=1))
    # breakpoint()

# breakpoint()
# base_backward_values = torch.cat(base_backward_values).reshape(-1, 50000)
base_backward_values = torch.cat(base_backward_values, dim=1).flip(0) # flip because backward pass reverse order
# base_logit_norm = torch.cat(base_logit_norm)
base_x_norm = torch.cat(base_x_norm)
torch.save(base_backward_values.cpu(), f'gradnorm/{model_name}_{dataset_name}_base_backward_values.pth')
# torch.save(base_logit_norm.cpu(), f'gradnorm/{model_name}_base_logit_norm.pth')
torch.save(base_x_norm.cpu(), f'gradnorm/{model_name}_{dataset_name}_base_x_norm.pth')

from tqdm import tqdm
from torch.autograd import Variable

backward_values = []
base_model = base_model.to(device).eval()
pert_feat_norm = []
pert_backward_values = []
pert_x_norm = []
for img, label in tqdm(loader):
    img = img.to(device)
    label = label.to(device)
    backward = []
    img = (img - mean) / std

    img = Variable(img.float(), requires_grad=True)

    logit = base_model(img)
    loss = margin_loss(logit, label)
    loss.sum().backward()
    # print(img.grad.norm())
    img = img + img.grad * 0.2
    
    img = Variable(img.float(), requires_grad=True)

    logit = base_model(img)
    loss = margin_loss(logit, label)
    loss.sum().backward(retain_graph=True)
    pert_backward_values.append(torch.stack(backward))
    pert_x_norm.append(img.grad.flatten(1).norm(dim=1))
    # base_feat_norm.append(feat.grad.flatten(1).norm(dim=1))
    # breakpoint()

# pert_backward_values = torch.cat(backward_values).reshape(-1, 50000)
pert_backward_values = torch.cat(pert_backward_values, dim=1).flip(0)

pert_x_norm = torch.cat(pert_x_norm)
torch.save(pert_backward_values.cpu(), f'gradnorm/{model_name}_{dataset_name}_pert_backward_values.pth')
torch.save(pert_x_norm.cpu(), f'gradnorm/{model_name}_{dataset_name}_pert_x_norm.pth')    
