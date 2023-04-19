import types
import torch
import torch.nn as nn
import timm

from .robust_vit import VisionTransformer
from .wrapper import ModelWrapper

class RandomDefense(nn.Module):
    def __init__(self, noise) -> None:
        super().__init__()
        self.noise = noise

    def forward(self, x):
        return x + self.noise * torch.randn_like(x)

def defense_token(x, defense_type, noise_sigma):
    if defense_type == 'gauss_filter':
        # gauss_x = gauss_filter(x.detach().cpu().numpy(), self.filter_sigma)
        gauss_x = torch.tensor(gauss_x).cuda()
        return gauss_x
    elif defense_type == 'random_noise':
        noise = torch.randn_like(x) * noise_sigma
        return x + noise 
    elif defense_type == 'laplace':
        d = torch.distributions.laplace.Laplace(torch.zeros_like(x), noise_sigma * torch.ones_like(x))
        return x + d.sample()
    elif defense_type == 'identical':
        return x

def add_defense(model_name, model, defense_type, def_position, noise, layer_index=-1):
    if isinstance(layer_index, int):
        layer_index = [layer_index]
    if 'resnet' in model_name:
        if def_position == 'hidden_feature':
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
                    x = defense_token(x, defense_type, noise)
                
                x = self.layer4(x)
                return x

            model.forward_features = types.MethodType(forward_features_new, model)
    if 'vgg' in model_name:
        if def_position == 'hidden_feature':
            def forward_features_new(self, x):
                for l in self.features:
                    if 'conv' in l._get_name().lower():
                        x = defense_token(x, defense_type, noise)
                    x = l(x)
                return x
            model.forward_features = types.MethodType(forward_features_new, model)
    if 'deit' in model_name:
        if def_position == 'hidden_feature':
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    if 'mixer' in model_name or 'resmlp' in model_name:
        if def_position == 'hidden_feature':
            model.blocks = nn.Sequential(*sum([[RandomDefense(noise), b] if i in layer_index or -1 in layer_index else [b] for i, b in enumerate(model.blocks)], []))
    if 'poolformer' in model_name:
        if def_position == 'hidden_feature':
            for i in range(len(model.network)):
                if type(model.network[i]).__name__ == 'Sequential':
                    model.network[i] = nn.Sequential(*sum([[RandomDefense(noise), b] if type(b).__name__ == 'PoolFormerBlock' else [b] for b in model.network[i] ], []))
    return model

def create_robust_model(model_name, dataset, n_cls, noise, defense, def_position, device='cpu', layer_index=-1):
    if 'vit' not in model_name and 'torchvision' not in model_name:
        base_model = timm.create_model(model_name, pretrained=False)
    elif 'vit' in model_name:
        if 'small' in model_name:
            kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
        else:
            kwargs = dict()
        base_model = VisionTransformer(weight_init='skip', num_classes=n_cls, defense_cls=defense, noise_sigma=noise, def_position=def_position, **kwargs)
        base_model.default_cfg = timm.create_model(model_name).default_cfg
        base_model.layer_index = layer_index
        base_model.set_defense(True)
    if dataset == 'cifar10':
        model_path = model_name.replace('.', '_')
        base_model.load_state_dict(torch.load(f'pretrain/{model_path}_{dataset}.pth.tar', map_location='cpu')['state_dict'])
    elif (dataset == 'imagenet' or dataset == 'imagenet_baseline') and 'torchvision' not in model_name:
        base_model.load_state_dict(timm.create_model(model_name, pretrained=True).state_dict())
            
    
    # base_model.layer_index = layer_index
    # base_model.set_defense(True)
    # base_model = torch.jit.trace(base_model, torch.randn(200, 3, 224, 224))
    if 'torchvision' in model_name:
        import torchvision
        base_model = torchvision.models.resnet50(pretrained=True)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = base_model.default_cfg['mean']
        std = base_model.default_cfg['std']
    base_model.noise_sigma = noise
    if 'vit' not in model_name and 'torchvision' not in model_name:
        base_model = add_defense(model_name, base_model, defense, def_position, noise, layer_index)
    model = ModelWrapper(base_model, num_classes=n_cls, device=device, def_position=def_position, mean=mean, std=std)
    return model