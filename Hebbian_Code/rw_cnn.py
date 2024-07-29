import torch
import torch.nn as nn
import torch.nn.functional as F


class RWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 mode='global', lambda_max=1.0, use_input_as_stimuli=True):
        super(RWConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.mode = mode
        self.lambda_max = lambda_max
        self.use_input_as_stimuli = use_input_as_stimuli

        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, *self.kernel_size))

        self.alpha_bp = 1
        self.alpha = nn.Parameter(torch.rand(out_channels), requires_grad=False)
        self.beta = nn.Parameter(torch.rand(out_channels), requires_grad=False)

        self.register_buffer('delta_w', torch.zeros_like(self.weight))

    def forward(self, x):
        constrained_weight = torch.sigmoid(self.weight)
        y = F.conv2d(x, constrained_weight, None, self.stride, self.padding)
        if self.training:
            self.compute_update(x, y)
        return y

    def compute_update(self, x, y):
        if self.mode == 'global':
            self.global_update(x, y)
        elif self.mode == 'local_within_filter':
            self.local_within_filter_update(x, y)
        elif self.mode == 'between_spatial_neurons':
            self.between_spatial_neurons_update(x, y)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def global_update(self, x, y):
        constrained_weight = torch.sigmoid(self.weight)
        batch_size, _, h_out, w_out = y.shape
        sum_V = y.view(batch_size, self.out_channels, -1).sum(dim=2)
        prediction_error = self.lambda_max - sum_V.mean(dim=0)
        prediction_error = prediction_error / (h_out * w_out * self.out_channels)

        stimuli = x if self.use_input_as_stimuli else y
        stimuli_unf = F.unfold(stimuli, self.kernel_size, stride=self.stride, padding=self.padding)
        y_unf = y.view(batch_size, self.out_channels, -1)
        correlation = torch.bmm(y_unf, stimuli_unf.transpose(1, 2)) / (h_out * w_out)

        delta_V = (self.alpha.view(1, -1, 1) * self.beta.view(1, -1, 1) * prediction_error.view(1, -1,
                                                                                                1) * correlation).mean(
            0)
        delta_V = delta_V.view(self.weight.shape)

        max_delta = torch.max(1 - constrained_weight, torch.zeros_like(constrained_weight))
        min_delta = torch.min(-constrained_weight, torch.zeros_like(constrained_weight))
        delta_V = torch.clamp(delta_V, min=min_delta, max=max_delta)

        self.delta_w += delta_V

    def local_within_filter_update(self, x, y):
        constrained_weight = torch.sigmoid(self.weight)
        batch_size, _, h_out, w_out = y.shape

        stimuli = x if self.use_input_as_stimuli else y
        stimuli_unf = F.unfold(stimuli, self.kernel_size, stride=self.stride, padding=self.padding)
        y_unf = y.view(batch_size, self.out_channels, -1)

        # Compute local prediction errors for each filter
        sum_V = y_unf.sum(dim=2)
        prediction_error = self.lambda_max - sum_V
        prediction_error = prediction_error / (h_out * w_out)

        correlation = torch.bmm(y_unf, stimuli_unf.transpose(1, 2)) / (h_out * w_out)

        delta_V = self.alpha.view(1, -1, 1) * self.beta.view(1, -1, 1) * prediction_error.unsqueeze(-1) * correlation
        delta_V = delta_V.mean(0).view(self.weight.shape)

        max_delta = torch.max(1 - constrained_weight, torch.zeros_like(constrained_weight))
        min_delta = torch.min(-constrained_weight, torch.zeros_like(constrained_weight))
        delta_V = torch.clamp(delta_V, min=min_delta, max=max_delta)

        self.delta_w += delta_V

    def between_spatial_neurons_update(self, x, y):
        constrained_weight = torch.sigmoid(self.weight)
        batch_size, _, h_out, w_out = y.shape

        stimuli = x if self.use_input_as_stimuli else y
        stimuli_unf = F.unfold(stimuli, self.kernel_size, stride=self.stride, padding=self.padding)
        y_unf = y.view(batch_size, self.out_channels, -1)

        # Compute prediction errors for each spatial location
        sum_V = y.sum(dim=1)  # Sum across channels
        prediction_error = self.lambda_max - sum_V
        prediction_error = prediction_error.view(batch_size, 1, -1) / self.out_channels

        correlation = torch.bmm(y_unf, stimuli_unf.transpose(1, 2))

        delta_V = self.alpha.view(1, -1, 1) * self.beta.view(1, -1, 1) * prediction_error * correlation
        delta_V = delta_V.mean(0).view(self.weight.shape)

        max_delta = torch.max(1 - constrained_weight, torch.zeros_like(constrained_weight))
        min_delta = torch.min(-constrained_weight, torch.zeros_like(constrained_weight))
        delta_V = torch.clamp(delta_V, min=min_delta, max=max_delta)

        self.delta_w += delta_V

    def rw_cnn_update(self, x, y):
        constrained_weight = torch.sigmoid(self.weight)
        batch_size, _, h_out, w_out = y.shape

        # Treat each spatial location as a separate "trial"
        y_spatial = y.view(batch_size, self.out_channels, -1)  # (batch, channels, spatial)

        # Total prediction for each spatial location
        total_prediction = y_spatial.sum(dim=1)  # (batch, spatial)

        # Shared prediction error for each spatial location
        prediction_error = self.lambda_max - total_prediction  # (batch, spatial)

        # Normalize prediction error
        prediction_error = prediction_error / self.out_channels

        # Compute salience (α) for each filter at each spatial location
        salience = y_spatial / (total_prediction.unsqueeze(1) + 1e-6)  # (batch, channels, spatial)

        # Compute update for each filter at each spatial location
        delta_V_spatial = salience * self.beta * prediction_error.unsqueeze(1)  # (batch, channels, spatial)

        # Reshape delta_V to match weight dimensions
        delta_V = delta_V_spatial.view(batch_size, self.out_channels, h_out, w_out)

        # Compute correlation with input
        if self.use_input_as_stimuli:
            x_unf = F.unfold(x, self.kernel_size, stride=self.stride, padding=self.padding)
            x_unf = x_unf.view(batch_size, self.in_channels, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
            delta_V = delta_V.view(batch_size, self.out_channels, 1, h_out * w_out)
            delta_W = torch.matmul(delta_V, x_unf.permute(0, 3, 1, 2)).mean(0)
            delta_W = delta_W.view(self.weight.shape)
        else:
            delta_W = F.conv2d(x.transpose(0, 1), delta_V.transpose(0, 1), padding=self.padding).transpose(0, 1)
            delta_W = delta_W.mean(0)

        # Ensure update is within bounds
        max_delta = torch.max(1 - constrained_weight, torch.zeros_like(constrained_weight))
        min_delta = torch.min(-constrained_weight, torch.zeros_like(constrained_weight))
        delta_W = torch.clamp(delta_W, min=min_delta, max=max_delta)

        self.delta_w += delta_W

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