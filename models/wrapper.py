import torch
import numpy as np
import utils
import math

class ModelWrapper:
    def __init__(self, model, num_classes=10, def_position=None, device='cpu', mean=None, std=None):
        self.model = model
        self.num_classes = num_classes
        self.model.to(device)
        self.batch_size = 128
        self.device = device
        # self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
        # self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        self.mean = mean if mean is not None else [0.5, 0.5, 0.5]
        self.std = std if std is not None else [0.5, 0.5, 0.5]
        self.mean = np.reshape(self.mean, [1, 3, 1, 1])
        self.std = np.reshape(self.std, [1, 3, 1, 1])
        self.mean_torch = torch.tensor(self.mean, device=device, dtype=torch.float32)
        self.std_torch = torch.tensor(self.std, device=device, dtype=torch.float32)
        self.def_position = def_position
        self.model.eval()
        # self.model.set_defense(True)
        # self.model = torch.jit.trace(self.model, torch.randn(self.batch_size, 3, 224, 224, device=device))

    def __call__(self, x):
        if self.def_position == 'input_noise':
            x = x + np.random.normal(scale=self.model.noise_sigma, size=x.shape).astype(np.float32)
        x = x.to(self.device)
        x = (x - self.mean_torch) / self.std_torch
        return self.model(x).cpu()

    def predict(self, x, return_tensor=False, defense=True):
        # self.model.set_defense(defense)
        # if self.def_position == 'input_noise' and defense:
        #     # max_norm = np.random.exponential(scale=1.0, size=None)
        #     # pert = np.random.uniform(-max_norm, max_norm, size=x.shape)
        #     # x = x + pert
        #     x = x + np.random.normal(scale=self.model.noise_sigma, size=x.shape)
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
                if self.def_position == 'input_noise' and defense:
                    x_batch_torch = x_batch_torch + self.model.noise_sigma * torch.randn_like(x_batch_torch) / self.std_torch
            # for x_batch in loader:
            #     x_batch_torch = x_batch.to(self.device)
                
                logits = self.model(x_batch_torch)[:, :self.num_classes].cpu()
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