from .base import BaseAttack
from .utils import *

import torch

class NES(BaseAttack):

    def __init__(self, model, log, eps, n_queries_each, sigma, step_size, targeted):
        super().__init__(model, log, 'l2', eps)
        self.n_queries_each = n_queries_each
        self.sigma = sigma
        self.targeted = targeted
        self.step_size = step_size

    @torch.no_grad()
    def run_one_iter(self, x, labels):
        num_dim = len(x.shape[1:])
        total_grad = torch.zeros_like(x)
        for _ in range(self.n_queries_each):
            tangent = torch.randn_like(x)
            forward_x = x + self.sigma * tangent
            backward_x = x - self.sigma * tangent
            forward_y = self.model.predict(forward_x, True)
            backward_y = self.model.predict(backward_x, True)
            change = (self.model.loss(forward_y, labels, targeted=self.targeted) - self.model.loss(backward_y, labels, targeted=self.targeted)) / (4 * self.sigma)
            total_grad += change.reshape(-1, *[1] * num_dim) * tangent

        new_x = x + self.step_size * total_grad / total_grad.norm(dim=list(range(1, num_dim+1)), keepdim=True)
        return new_x, 2 * self.n_queries_each * torch.ones(x.shape[0])

    def attack(self, x, labels, n_iter):
        total_queries = 0
        for i in range(n_iter):
            x, n_queries = self.run_one_iter(x, labels)
            pred = self.get_pred(x)
