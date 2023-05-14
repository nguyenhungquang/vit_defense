import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
import math

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.from_numpy(mean))
        self.std = nn.Parameter(torch.from_numpy(std))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x

class ModelWrapper(nn.Module):
    def __init__(self, model, num_classes=10, def_position=None, device='cpu', mean=None, std=None):
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = 64
        self.device = device
        # self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
        # self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        mean = mean if mean is not None else [0.5, 0.5, 0.5]
        std = std if std is not None else [0.5, 0.5, 0.5]
        mean = np.reshape(mean, [1, 3, 1, 1]).astype(np.single)
        std = np.reshape(std, [1, 3, 1, 1]).astype(np.single)
        # self.mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        # self.std_torch = torch.tensor(self.std, device=device, dtype=torch.float32)
        self.def_position = def_position
        self.model = nn.Sequential(Normalization(mean, std), model)
        self.model.noise_sigma = model.noise_sigma
        self.model.to(device)
        self.model.eval()
        if def_position == 'aaa_linear':
            self.attractor_interval = 6
            self.reverse_step = 1
            self.calibration_loss_weight = 5
            self.aaa_iter = 100
            self.aaa_optimizer_lr = 0.1
            self.temperature = 1#1.1236
            self.dev = 0.5
        # self.model.set_defense(True)
        # self.model = torch.jit.trace(self.model, torch.randn(self.batch_size, 3, 224, 224, device=device))

    def aaa_forward(self, x):
        with torch.no_grad():
            logits = self.model(x)

        logits_ori = logits.detach()
        prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
        prob_max_ori = prob_ori.max(1)[0] ###
        value, index_ori = torch.topk(logits_ori, k=2, dim=1)
        #"""
        mask_first = torch.zeros(logits.shape, device=self.device)
        mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
        mask_second = torch.zeros(logits.shape, device=self.device)
        mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
        #"""
        
        margin_ori = value[:, 0] - value[:, 1]
        attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
        target = attractor - self.reverse_step * (margin_ori - attractor)
        # diff_ori = (margin_ori - target)
        # real_diff_ori = margin_ori - attractor

        with torch.enable_grad():
            logits.requires_grad = True
            optimizer = torch.optim.Adam([logits], lr=self.aaa_optimizer_lr)
            i = 0 
            # los_reverse_rate = 0
            # prd_maintain_rate = 0
            for i in range(self.aaa_iter):
            #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                prob = F.softmax(logits, dim=1)
                #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                #loss_calibration = (prob - prob_ori).abs().mean()

                value, index = torch.topk(logits, k=2, dim=1) 
                margin = value[:, 0] - value[:, 1]
                #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                diff = (margin - target)
                real_diff = margin - attractor
                loss_defense = diff.abs().mean()
                
                loss = loss_defense + loss_calibration * self.calibration_loss_weight
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return logits.detach()

    def forward(self, x):
        # if self.def_position == 'input_noise':
        #     x = x + np.random.normal(scale=self.model.noise_sigma, size=x.shape).astype(np.float32)
        x = x.to(self.device)
        if self.def_position == 'input_noise':
            x = x + self.model.noise_sigma * torch.randn_like(x)
        # x = (x - self.mean_torch) / self.std_torch
        if self.def_position == 'aaa_linear':
            out = self.aaa_forward(x)
        else:
            out = self.model(x)
        if self.def_position == 'logits':
            out = out + self.model.noise_sigma * torch.randn_like(out)
        return out#.cpu()

    def predict(self, x, return_tensor=False, defense=True):
        # self.model.set_defense(defense)
        # if self.def_position == 'input_noise' and defense:
        #     # max_norm = np.random.exponential(scale=1.0, size=None)
        #     # pert = np.random.uniform(-max_norm, max_norm, size=x.shape)
        #     # x = x + pert
        #     x = x + np.random.normal(scale=self.model.noise_sigma, size=x.shape)
        # x = (x - self.mean) / self.std
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
                if self.def_position == 'input_noise' and defense:
                    x_batch_torch = x_batch_torch + self.model.noise_sigma * torch.randn_like(x_batch_torch) 
                # x_batch_torch = (x_batch_torch - self.mean_torch) / self.std_torch
            # for x_batch in loader:
            #     x_batch_torch = x_batch.to(self.device)
                if self.def_position == 'aaa_linear':
                    logits = self.aaa_forward(x_batch_torch)[:, :self.num_classes]
                else:
                    logits = self.model(x_batch_torch)[:, :self.num_classes]
                if self.def_position == 'logits':
                    logits = logits + self.model.noise_sigma * torch.randn_like(logits)
                logits = logits.cpu()
                if not return_tensor:
                    logits = logits.numpy()
                logits_list.append(logits)
        # self.model.set_defense(True)
        if return_tensor:
            logits = torch.cat(logits_list)
        else:
            logits = np.vstack(logits_list)
        return logits

    def loss(self, y, logits, targeted=False, loss_type='margin_loss'):
        y = utils.random_classes_except_current(y, self.num_classes) if targeted else y
        y = utils.dense_to_onehot(y, n_cls=self.num_classes)
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        if torch.is_tensor(logits):
            logits = logits.detach().cpu().numpy()
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