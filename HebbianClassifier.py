import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianClassifier(nn.Module):
    def __init__(self, in_features, out_features, lr=0.01, mode='oja'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.lr = lr
        self.mode = mode
        self.register_buffer('delta_w', torch.zeros_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight)

    def update(self, x, y):
        with torch.no_grad():
            if self.mode == 'oja':
                # Oja's rule
                hebb_term = torch.einsum('bi,bj->ij', y, x)
                norm_term = (y ** 2).sum(dim=0, keepdim=True) * self.weight
                self.delta_w = self.lr * (hebb_term - norm_term)
            elif self.mode == 'hebb':
                # Basic Hebbian rule
                self.delta_w = self.lr * torch.einsum('bi,bj->ij', y, x)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

    def apply_update(self):
        with torch.no_grad():
            self.weight += self.delta_w
            self.delta_w.zero_()