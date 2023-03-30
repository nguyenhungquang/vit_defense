import types
import torch
import timm

from .robust_vit import VisionTransformer
from .wrapper import ModelWrapper

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

def add_defense(model_name, model, defense_type, def_position, noise):
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
                x = defense_token(x, defense_type, noise)
                x = self.layer1(x)
                x = defense_token(x, defense_type, noise)
                x = self.layer2(x)
                x = defense_token(x, defense_type, noise)
                x = self.layer3(x)
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
    return model

def create_model(model_name, dataset, n_cls, noise, defense, def_position, device='cpu'):
    if 'vit' not in model_name:
        base_model = timm.create_model(model_name, pretrained=False)
        base_model = add_defense(model_name, base_model, defense, def_position, noise)
    elif 'vit' in model_name:
        if 'small' in model_name:
            kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
        else:
            kwargs = dict()
        base_model = VisionTransformer(weight_init='skip', num_classes=n_cls, defense_cls=defense, noise_sigma=noise, def_position=def_position, **kwargs)
        base_model.default_cfg = timm.create_model(model_name).default_cfg
        base_model.layer_index = -1
        base_model.set_defense(True)
    if dataset == 'cifar10':
        base_model.load_state_dict(torch.load(f'pretrain/{model_name}_{dataset}.pth.tar', map_location='cpu')['state_dict'])
    elif dataset == 'imagenet':
        base_model.load_state_dict(timm.create_model(model_name, pretrained=True).state_dict())
    base_model.noise_sigma = noise
    # base_model.layer_index = layer_index
    # base_model.set_defense(True)
    # base_model = torch.jit.trace(base_model, torch.randn(200, 3, 224, 224))
    model = ModelWrapper(base_model, num_classes=n_cls, device=device, def_position=def_position, mean=base_model.default_cfg['mean'], std=base_model.default_cfg['std'])
    return model