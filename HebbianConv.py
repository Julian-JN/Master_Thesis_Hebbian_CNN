import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def normalize(x, dim=None):
    """
    Normalize the input tensor along the specified dimension.
    """
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


class HebbianConv2d(nn.Module):
    """
    A 2D convolutional layer that learns through Hebbian plasticity.
    """

    MODE_WTA = 'wta'
    MODE_OJA = 'oja'
    MODE_BASIC = 'basic'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 w_nrm=True, bias=False, act=nn.Identity(),
                 mode=MODE_WTA, alpha=1.0, lr=0.01):
        """
        Initialize the HebbianConv2d layer.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Size of the convolutional kernel (int or tuple).
        :param stride: Stride of the convolutional kernel (int or tuple).
        :param w_nrm: Whether to normalize the weight vectors before computing outputs.
        :param bias: Whether to include a bias term.
        :param act: Nonlinear activation function after convolution.
        :param mode: Learning mode, can be 'wta', 'oja', or 'basic'.
        :param alpha: Weighting coefficient between Hebbian and backprop updates.
        :param lr: Learning rate for the Hebbian update rule.
        """
        super().__init__()
        self.mode = mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        # Initialize weights
        self.weight = nn.Parameter(torch.empty((self.out_channels, self.in_channels, *self.kernel_size)),
                                   requires_grad=True)
        nn.init.xavier_normal_(self.weight)
        self.w_nrm = w_nrm
        self.bias = nn.Parameter(torch.zeros(self.out_channels), requires_grad=bias)
        self.act = act
        self.register_buffer('delta_w', torch.zeros_like(self.weight))

        self.alpha = alpha
        self.lr = lr

    def forward(self, x):
        """
        Forward pass for the HebbianConv2d layer.

        :param x: Input tensor.
        :return: Output tensor after convolution and activation.
        """
        if self.w_nrm:
            weight = normalize(self.weight, dim=(1, 2, 3))
        else:
            weight = self.weight

        y = F.conv2d(x, weight, self.bias, stride=self.stride)
        y = self.act(y)

        if self.training:
            self._hebbian_update(x, y)

        return y

    def _hebbian_update(self, x, y):
        """
        Update weights based on Hebbian learning rule.

        :param x: Input tensor.
        :param y: Output tensor.
        """
        with torch.no_grad():
            x_patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
            x_patches = x_patches.view(x.size(0), self.in_channels, *self.kernel_size, -1)
            y_flat = y.view(y.size(0), self.out_channels, -1)

            if self.mode == self.MODE_WTA:
                # Winner-Take-All (WTA) Hebbian update rule
                wta_mask = torch.zeros_like(y_flat)
                wta_mask.scatter_(1, y_flat.argmax(dim=1, keepdim=True), 1)
                hebb_update = torch.einsum('bclij,bkc->kcij', x_patches, wta_mask)
                weight_update = hebb_update.sum(dim=0) / (wta_mask.sum() + 1e-8)

            elif self.mode == self.MODE_BASIC:
                # Basic Hebbian update rule
                hebb_update = torch.einsum('bclij,bkc->kcij', x_patches, y_flat)
                weight_update = self.lr * hebb_update.mean(dim=0)

            elif self.mode == self.MODE_OJA:
                # Oja's Hebbian update rule
                hebb_update = torch.einsum('bclij,bkc->kcij', x_patches, y_flat)
                normalization = y_flat.pow(2).sum(dim=2, keepdim=True)
                weight_update = self.lr * (hebb_update - self.weight.unsqueeze(3).unsqueeze(4) * normalization.unsqueeze(2).unsqueeze(3)).mean(dim=0)

            self.delta_w += weight_update

    @torch.no_grad()
    def local_update(self):
        """
        Transfer the computed weight update to the gradient for the optimizer.

        This function should be called before optimizer.step(), so that the optimizer will use the locally computed
        update as the optimization direction. Local updates can also be combined with end-to-end updates by calling
        this function between loss.backward() and optimizer.step().
        """
        if self.weight.grad is None:
            self.weight.grad = -self.alpha * self.delta_w
        else:
            self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
        self.delta_w.zero_()
