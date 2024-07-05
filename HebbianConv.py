import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def normalize(x, dim=None):
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


class HebbianConv2d(nn.Module):
    """
    A 2d convolutional layer that learns through Hebbian plasticity
    """

    MODE_WTA = 'wta'
    MODE_OJA = 'oja'
    MODE_BASIC = 'basic'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 w_nrm=True, bias=False, act=nn.Identity(),
                 mode=MODE_WTA, alpha=1.0, lr=0.01):
        """

        :param out_channels: output channels of the convolutional kernel
        :param in_channels: input channels of the convolutional kernel
        :param kernel_size: size of the convolutional kernel (int or tuple)
        :param stride: stride of the convolutional kernel (int or tuple)
        :param w_nrm: whether to normalize the weight vectors before computing outputs
        :param act: the nonlinear activation function after convolution
        :param mode: the learning mode, can be 'wta', 'oja', or 'basic'
        :param alpha: weighting coefficient between hebbian and backprop updates (0 means fully backprop, 1 means fully hebbian).
        :param lr: learning rate for the Hebbian update rule

        """

        super().__init__()
        self.mode = mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        self.weight = nn.Parameter(torch.empty((self.out_channels, self.in_channels, *self.kernel_size)),
                                   requires_grad=True)
        nn.init.xavier_normal_(self.weight)
        self.w_nrm = w_nrm
        self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=bias)
        self.act = act
        self.register_buffer('delta_w', torch.zeros_like(self.weight))

        self.alpha = alpha
        self.lr = lr  # Learning rate for Hebbian update

    def apply_weights(self, x, w):
        """
        This function provides the logic for combining input x and weight w
        """

        return torch.conv2d(x, w, bias=self.bias, stride=self.stride)

    def compute_activation(self, x):
        w = self.weight
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        y = self.act(self.apply_weights(x, w))
        return y

    def forward(self, x):
        print("Input")
        print(x.shape)
        y = self.compute_activation(x)
        print(y.shape)
        if self.training and self.alpha != 0: self.compute_update(x, y)
        return y

    def compute_update(self, x, y):
        if self.mode not in [self.MODE_WTA, self.MODE_OJA, self.MODE_BASIC]:
            raise NotImplementedError(
                f"Learning mode {self.mode} unavailable for {self.__class__.__name__} layer")

        print("Update")
        with torch.no_grad():
            # Unfold input to patches
            x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
            x_unf = x_unf.permute(0, 2, 1)  # (batch, L, C*kH*kW)
            print(x_unf.shape)

            # Pre-synaptic activations (input patches)
            x_patches = x_unf.reshape(x.size(0), -1, self.in_channels, *self.kernel_size)

            # Post-synaptic activations (output of the layer)
            y_flat = y.reshape(y.size(0), y.size(1), -1)  # (batch, out_channels, L)
            print(y_flat.shape)

            if self.mode == self.MODE_WTA:
                # Winner-Take-All competition
                max_activations = y_flat.argmax(dim=1, keepdim=True)  # Indices of maximum activations
                wta_mask = torch.zeros_like(y_flat).scatter_(1, max_activations, 1.0)  # Binary mask
                print(wta_mask.shape)
                print(x_patches.shape)
                # Basic Hebbian update rule
                hebb_update = torch.einsum('bkl,bclij->bkcij', wta_mask, x_patches)
                weight_update = hebb_update.sum(dim=0) / (wta_mask.sum() + 1e-8)  # Weighted mean update

            elif self.mode == self.MODE_BASIC:
                # Basic Hebbian update rule (without WTA)
                hebb_update = torch.einsum('bkl,bclij->kcij', y_flat, x_patches)
                weight_update = self.lr * hebb_update

            elif self.mode == self.MODE_OJA:
                # Oja's rule
                hebb_update = torch.einsum('bkl,bclij->kcij', y_flat, x_patches)  # Hebbian term
                normalization = y_flat.pow(2).sum(dim=2, keepdim=True)  # Normalization term
                weight_update = self.lr * (
                            hebb_update - self.weight.unsqueeze(3).unsqueeze(4) * normalization.unsqueeze(2).unsqueeze(
                        3))

            self.delta_w += weight_update  # Accumulate updates

    @torch.no_grad()
    def local_update(self):
        """
        This function transfers a previously computed weight update, stored in buffer self.delta_w, to the gradient
        self.weight.grad of the weight parameter.

        This function should be called before optimizer.step(), so that the optimizer will use the locally computed
        update as optimization direction. Local updates can also be combined with end-to-end updates by calling this
        function between loss.backward() and optimizer.step(). loss.backward will store the end-to-end gradient in
        self.weight.grad, and this function combines this value with self.delta_w as
        self.weight.grad = (1 - alpha) * self.weight.grad - alpha * self.delta_w
        Parameter alpha determines the scale of the local update compared to the end-to-end gradient in the combination.

        """

        if self.weight.grad is None:
            self.weight.grad = -self.alpha * self.delta_w
        else:
            self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
        self.delta_w.zero_()
