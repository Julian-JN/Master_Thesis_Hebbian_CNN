import torch
import torch.nn as nn
import torch.nn.functional as F


class RWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 mode='unsupervised', lambda_max=1.0, use_input_as_stimuli=True):
        super(RWConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.mode = mode
        self.lambda_max = lambda_max
        self.use_input_as_stimuli = use_input_as_stimuli

        # Initialize weights between 0 and 1
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, *self.kernel_size))

        self.alpha_bp = 1
        self.alpha = nn.Parameter(torch.rand(out_channels), requires_grad=False)
        self.beta = nn.Parameter(torch.rand(out_channels), requires_grad=False)
        print(f"Alpha Max and Min: {self.alpha.max(), self.alpha.min()}")

        self.register_buffer('delta_w', torch.zeros_like(self.weight))

    def forward(self, x):
        # Apply sigmoid to constrain weights between 0 and 1
        constrained_weight = torch.sigmoid(self.weight)
        y = F.conv2d(x, constrained_weight, None, self.stride, self.padding)
        if self.training:
            self.compute_update(x, y)
        return y

    def compute_update(self, x, y):
        if self.mode == 'unsupervised':
            self.unsupervised_update(x, y)
        elif self.mode == 'supervised':
            raise NotImplementedError("Supervised mode not implemented in this example")

    def unsupervised_update(self, x, y):
        constrained_weight = torch.sigmoid(self.weight)
        # constrained_weight = self.weight
        print(f"Weights Max, Min and Mean: {constrained_weight.max(), constrained_weight.min(), constrained_weight.mean()}")
        batch_size, _, h_out, w_out = y.shape
        sum_V = y.sum()
        prediction_error = self.lambda_max - sum_V
        stimuli = x if self.use_input_as_stimuli else y
        stimuli_unf = F.unfold(stimuli, self.kernel_size, stride=self.stride, padding=self.padding)
        y_unf = y.view(batch_size, self.out_channels, -1)

        correlation = torch.bmm(y_unf, stimuli_unf.transpose(1, 2))
        # Normalize prediction error to prevent large uniform updates
        prediction_error = prediction_error / (h_out * w_out * self.out_channels)
        # Compute update for all filters at once
        delta_V = (self.alpha.view(1, -1, 1) * self.beta.view(1, -1, 1) * prediction_error * correlation).mean(0)
        delta_V = delta_V.view(self.weight.shape)

        # # Ensure update is within bounds
        max_delta = torch.max(1 - constrained_weight, torch.zeros_like(constrained_weight))
        min_delta = torch.min(-constrained_weight, torch.zeros_like(constrained_weight))
        delta_V = torch.clamp(delta_V, min=min_delta, max=max_delta)

        print(f"delta_V shape: {delta_V.shape}")
        print(f"delta_V Weights Max, Min and Mean: {delta_V.max(), delta_V.min(), delta_V.mean()}")

        self.delta_w += delta_V

    @torch.no_grad()
    def local_update(self):
        if self.weight.grad is None:
            self.weight.grad = -self.alpha_bp * self.delta_w
        else:
            self.weight.grad = (1 - self.alpha_bp) * self.weight.grad - self.alpha_bp * self.delta_w
        self.delta_w.zero_()

    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
               f'kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, ' \
               f'mode={self.mode}, lambda_max={self.lambda_max}, ' \
               f'use_input_as_stimuli={self.use_input_as_stimuli}'