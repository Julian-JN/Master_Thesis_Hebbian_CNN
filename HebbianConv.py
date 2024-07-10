import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


class HebbianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, learning_rate=0.01, mode='hebbian',
                 wta_competition='filter'):
        super(HebbianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.learning_rate = learning_rate
        self.mode = mode
        self.wta_competition = wta_competition
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size)).to(device)
        self.delta_w = torch.zeros_like(self.weight).to(device)

    def forward(self, x):
        y = F.conv2d(x, self.weight, stride=self.stride)
        y = F.relu(y)  # Example activation function
        return y

    def compute_update(self, x_unf, y_unf):
        if self.mode == 'hebbian':
            return y_unf.t().matmul(x_unf).reshape_as(self.weight)
        elif self.mode == 'oja':
            return (y_unf.t().matmul(x_unf) - y_unf.t().matmul(y_unf).matmul(
                self.weight.view(self.out_channels, -1))).reshape_as(self.weight)
        elif self.mode == 'wta':
            if self.wta_competition == 'filter':
                y_max, _ = y_unf.max(dim=1, keepdim=True)
                y_unf = (y_unf == y_max).float() * y_unf
            elif self.wta_competition == 'spatial':
                y_max, _ = y_unf.max(dim=0, keepdim=True)
                y_unf = (y_unf == y_max).float() * y_unf

            y_weighted = y_unf / (y_unf.sum(dim=0, keepdim=True) + 1e-6)
            return y_weighted.t().matmul(x_unf).reshape_as(self.weight)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def update_weights(self, x, y):
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
        y_unf = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
        self.delta_w += self.compute_update(x_unf, y_unf)
        self.weight.data += self.learning_rate * self.delta_w
        self.delta_w.zero_()
