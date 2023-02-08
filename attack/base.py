import torch
import torch.nn.functional as F
import math

class BaseAttack:
    def __init__(self, model, log, lp, eps):
        self.lp = lp
        self.eps = eps
        self.log = log
        self.model = model

    @torch.no_grad()
    def get_prob(self, x, y):
        probs = []
        bsz = 64
        n_batches = math.ceil(x.shape[0] / 64)
        for i in range(n_batches):
            x_batch = x[i*bsz:(i+1)*bsz]
            y_batch = y[i*bsz:(i+1)*bsz]
            logits = self.model(x_batch)
            prob = logits.softmax(dim=1)
            # out = self.model(x).softmax(dim=-1)
            prob = torch.gather(prob, 1, y_batch.unsqueeze(1)).squeeze(1)
            probs.append(prob)
        # prob = torch.index_select(F.softmax(out, dim=-1), 1, y).cpu()
        return torch.cat(probs)

    @torch.no_grad()
    def get_pred(self, x):
        preds = []
        bsz = 64
        n_batches = math.ceil(x.shape[0] / 64)
        for i in range(n_batches):
            x_batch = x[i*bsz:(i+1)*bsz]
            logits = self.model(x_batch)
            preds.append(logits.max(1)[1])
        return torch.cat(preds)
        # prob = torch.index_select(F.softmax(out, dim=-1), 1, y).cpu()
        # out = self.model.predict(x)
        # return out.max(1)[1]