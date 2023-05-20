import numpy as np
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .robust_vit import VisionTransformer
from .wrapper import ModelWrapper
from .noise_resnet import noise_resnet20
from .vanilla_resnet import vanilla_resnet20
# from utils_.utils import AverageMeter, RecorderMeter, time_string, convert_secs2time

class RandomDefense(nn.Module):
    def __init__(self, noise, scale=None) -> None:
        super().__init__()
        self.noise = noise
        # self.scale = nn.Parameter(torch.from_numpy(scale))
        if scale is not None:
            self.scale = nn.Parameter(scale)
        else:
            self.scale = None

    def forward(self, x):
        if self.scale is not None:
            out = x + self.noise * torch.randn_like(x) * self.scale#.to(x)
        else:
            out = x + self.noise * torch.randn_like(x)
        return out

def defense_token(x, defense_type, noise_sigma, scale=None):
    if defense_type == 'gauss_filter':
        # gauss_x = gauss_filter(x.detach().cpu().numpy(), self.filter_sigma)
        gauss_x = torch.tensor(gauss_x).cuda()
        return gauss_x
    elif defense_type == 'random_noise':
        noise = torch.randn_like(x) * noise_sigma
        if scale is not None:
            noise = noise * scale.to(x)
        return x + noise 
    elif defense_type == 'laplace':
        d = torch.distributions.laplace.Laplace(torch.zeros_like(x), noise_sigma * torch.ones_like(x))
        return x + d.sample()
    elif defense_type == 'identical':
        return x

def add_defense(model_name, model, defense_type, def_position, noise, layer_index=-1, scale=False, dset_name=None):
    if isinstance(layer_index, int):
        layer_index = [layer_index]
    if 'resnet' in model_name:
        if def_position == 'hidden_feature':
            if scale:
                std = torch.load(f'stats/{model_name}_{dset_name}_last_std.pth')
            def forward_features_new(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.act1(x)
                x = self.maxpool(x)

                # if self.grad_checkpointing and not torch.jit.is_scripting():
                #     x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4], x, flatten=True)
                # else:
                if 0 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer1(x)
                if 1 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer2(x)
                if 2 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer3(x)
                if 3 in layer_index or -1 in layer_index:
                    x = defense_token(x, defense_type, noise, scale=std if scale else None)
                
                x = self.layer4(x)
                return x

            model.forward_features = types.MethodType(forward_features_new, model)
    if 'vgg' in model_name:
        # with open(f'stats/{model_name}_cifar10.npy', 'rb') as fw:
        #     mean = np.load(fw, allow_pickle=True)
        # mean = mean / mean.mean()
        # print(layer_index)
        # with open(f'stats/{model_name}_cifar10_last_std.npy', 'rb') as fw:
        #     std = np.load(fw, allow_pickle=True).reshape(1, 512, 14, 14)
        if def_position == 'hidden_feature':
            
            def forward_features_new(self, x):
                # print(len([l for l in self.features if 'conv' in l._get_name().lower()]))
                c = 0
                for l in self.features: 
                    if 'conv' in l._get_name().lower():
                        if c in layer_index or layer_index == [-1]:
                            x = defense_token(x, defense_type, noise)
                        # if c == 15:
                        #     x = defense_token(x, defense_type, noise, std)
                        c += 1
                    x = l(x)
                return x
            model.forward_features = types.MethodType(forward_features_new, model)
    if 'vit' in model_name:
        if def_position == 'hidden_feature':
            # with open(f'stats/{model_name}_cifar10_last_std.npy', 'rb') as fw:
            #     std = np.load(fw, allow_pickle=True)
            std = torch.load(f'stats/{model_name}_cifar10_all_std.pth')
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise, std[i] if scale else None), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    if 'deit' in model_name:
        if def_position == 'hidden_feature':
            # with open(f'stats/{model_name}_cifar10_last_std.npy', 'rb') as fw:
            #     std = np.load(fw, allow_pickle=True)
            std = torch.load(f'stats/{model_name}_cifar10_all_std.pth')
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise, std[i] if scale else None), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    if 'mixer' in model_name or 'resmlp' in model_name:
        if def_position == 'hidden_feature':
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    if 'poolformer' in model_name:
        if def_position == 'hidden_feature':
            for i in range(len(model.network)):
                if type(model.network[i]).__name__ == 'Sequential':
                    model.network[i] = nn.Sequential(*sum([[RandomDefense(noise), b] if type(b).__name__ == 'PoolFormerBlock' else [b] for b in model.network[i] ], []))
    if 'vanilla_resnet' in model_name:
        if def_position == 'hidden_feature':
            def forward_new(self, x):
                x = self.conv_1_3x3(x)
                x = F.relu(self.bn_1(x), inplace=True)
                x = self.stage_1(x)
                x = defense_token(x, defense_type, noise)
                x = self.stage_2(x)
                x = defense_token(x, defense_type, noise)
                x = self.stage_3(x)
                x = defense_token(x, defense_type, noise)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
            model.forward = types.MethodType(forward_new, model)
    return model

def create_robust_model(model_name, dataset, n_cls, noise, defense, def_position, device='cpu', layer_index=-1, blackbox=True, scale=False):
    if 'noise_resnet' in model_name:
        assert dataset == 'cifar10'
        base_model = noise_resnet20()
    elif 'vanilla_resnet' in model_name:
        assert dataset == 'cifar10'
        base_model = vanilla_resnet20()
    elif 'vit' not in model_name and 'torchvision' not in model_name:
        base_model = timm.create_model(model_name, pretrained=False)
    elif 'vit' in model_name:
        if 'small' in model_name:
            kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
        else:
            kwargs = dict()
        base_model = VisionTransformer(weight_init='skip', num_classes=1000, defense_cls=defense, noise_sigma=noise, def_position=def_position, **kwargs)
        base_model.default_cfg = timm.create_model(model_name).default_cfg
        base_model.layer_index = layer_index
        base_model.set_defense(True)

    if dataset == 'cifar10':
        # breakpoint()
        model_path = model_name.replace('.', '_')
        # path = '../blackbox/CVPR_2019_PNI/code/save/2023-05-04/cifar10_vanilla_resnet20_160_SGD_train_channelwise_3e-4decay/model_best.pth.tar'
        state_dict = torch.load(f'pretrain/{model_path}_{dataset}.pth.tar', map_location='cpu')['state_dict']
        # state_dict = torch.load(path, map_location='cpu')['state_dict']
        if 'noise_resnet' in model_name or model_name == 'vanilla_resnet_adv':
            # state_tmp = base_model.state_dict()
            # state_tmp.update(state_dict)
            state_dict = {k[2:]:v for (k, v) in state_dict.items() if k[0] != '0'}
            # print(state_dict)
        base_model.load_state_dict(state_dict)
    elif (dataset == 'imagenet' or dataset == 'imagenet_baseline') and 'torchvision' not in model_name:
        base_model.load_state_dict(timm.create_model(model_name, pretrained=True).state_dict())
            
    
    # base_model.layer_index = layer_index
    # base_model.set_defense(True)
    # base_model = torch.jit.trace(base_model, torch.randn(200, 3, 224, 224))
    if 'noise_resnet' in model_name or 'vanilla_resnet' in model_name:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif 'torchvision' in model_name:
        import torchvision
        base_model = torchvision.models.resnet50(pretrained=True)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = base_model.default_cfg['mean']
        std = base_model.default_cfg['std']
    base_model.noise_sigma = noise
    # if 'vit' not in model_name and 'torchvision' not in model_name:
    if 'torchvision' not in model_name:
        base_model = add_defense(model_name, base_model, defense, def_position, noise, layer_index, scale=scale, dset_name=dataset)
    # from foolbox import PyTorchModel
    # preprocessing = dict(mean=mean, std=std, axis=-3)
    # base_model.eval()
    # model = PyTorchModel(base_model, bounds=(0, 1), preprocessing=preprocessing)
    # return model
    if blackbox:
        model = ModelWrapper(base_model, num_classes=n_cls, device=device, def_position=def_position, mean=mean, std=std)
        return model
    else:
        return base_model, mean, std