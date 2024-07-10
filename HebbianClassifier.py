import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class LinearHebbianClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, learning_rate=0.01, mode='hebbian', wta_competition='filter'):
        super(LinearHebbianClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.mode = mode
        self.wta_competition = wta_competition
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim)).to(device)
        self.delta_w = torch.zeros_like(self.weight).to(device)

    def forward(self, x):
        y = x.matmul(self.weight.t())
        y = F.relu(y)  # Example activation function
        return y

    def compute_update(self, x, y):
        if self.mode == 'hebbian':
            return y.t().matmul(x)
        elif self.mode == 'oja':
            return y.t().matmul(x) - y.t().matmul(y).matmul(self.weight)
        elif self.mode == 'wta':
            if self.wta_competition == 'filter':
                y_max, _ = y.max(dim=1, keepdim=True)
                y_unf = (y == y_max).float() * y
            elif self.wta_competition == 'spatial':
                y_max, _ = y.max(dim=0, keepdim=True)
                y_unf = (y == y_max).float() * y

            y_weighted = y_unf / (y_unf.sum(dim=0, keepdim=True) + 1e-6)
            return y_weighted.t().matmul(x)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def update_weights(self, x, y):
        self.delta_w += self.compute_update(x, y)
        self.weight.data += self.learning_rate * self.delta_w
        self.delta_w.zero_()
