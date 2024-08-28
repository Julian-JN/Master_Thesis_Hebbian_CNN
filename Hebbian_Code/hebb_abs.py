import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt

import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


def normalize(x, dim=None):
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


def symmetric_pad(x, padding):
    """
    Performs reflective padding on the 4D input tensor x.

    Args:
        x (torch.Tensor): 4D input tensor to be padded (batch, channels, height, width).
        padding (int): Amount of padding to be applied on each side.

    Returns:
        torch.Tensor: Padded input tensor.
    """
    if padding == 0:
        return x

    batch_size, channels, height, width = x.size()

    # Determine left, right, top, and bottom padding
    left_pad = right_pad = padding
    top_pad = bottom_pad = padding

    # Adjust right and bottom padding if necessary to maintain symmetry
    if width % 2 != 0:
        right_pad += 1
    if height % 2 != 0:
        bottom_pad += 1

    # Perform reflective padding
    padded_x = torch.cat((
        x[:, :, :, :padding].flip(dims=[-1]),  # Left padding
        x,
        x[:, :, :, -padding:].flip(dims=[-1])  # Right padding
    ), dim=-1)

    padded_x = torch.cat((
        padded_x[:, :, :padding, :].flip(dims=[-2]),  # Top padding
        padded_x,
        padded_x[:, :, -padding:, :].flip(dims=[-2])  # Bottom padding
    ), dim=-2)
    return padded_x


def create_sm_kernel(kernel_size=5, sigma_e=1.2, sigma_i=1.4):
    """
    Create a surround modulation kernel that matches the paper's specifications.

    :param kernel_size: Size of the SM kernel.
    :param sigma_e: Standard deviation for the excitatory Gaussian.
    :param sigma_i: Standard deviation for the inhibitory Gaussian.
    :return: A normalized SM kernel.
    """
    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij")
    x = x.float() - center
    y = y.float() - center

    # Compute the excitatory and inhibitory Gaussians
    gaussian_e = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_e ** 2))
    gaussian_i = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_i ** 2))

    # Compute the Difference of Gaussians (DoG)
    dog = gaussian_e / (2 * math.pi * sigma_e ** 2) - gaussian_i / (2 * math.pi * sigma_i ** 2)

    # Normalize the DoG so that the center value is 1
    sm_kernel = dog / dog[center, center]

    return sm_kernel.unsqueeze(0).unsqueeze(0).to(device)


class HebbianConv2d(nn.Module):
    """
	A 2d convolutional layer that learns through Hebbian plasticity
	"""

    MODE_HPCA = 'hpca'
    MODE_BASIC_HEBBIAN = 'basic'
    MODE_WTA = 'wta'
    MODE_SOFTWTA = 'soft'
    MODE_BCM = 'bcm'
    MODE_HARDWT = "hard"
    MODE_PRESYNAPTIC_COMPETITION = "pre"
    MODE_TEMPORAL_COMPETITION = "temp"
    MODE_ADAPTIVE_THRESHOLD = "thresh"
    MODE_ANTIHARDWT = "antihard"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, groups=1,
                 w_nrm=False, bias=False, act=nn.Identity(),
                 mode=MODE_SOFTWTA, k=1, patchwise=True,
                 contrast=1., uniformity=False, alpha=1., wta_competition='similarity_spatial',
                 lateral_competition="filter",
                 lateral_inhibition_strength=0.01, top_k=1, prune_rate=0, t_invert=1.):
        """

		:param out_channels: output channels of the convolutional kernel
		:param in_channels: input channels of the convolutional kernel
		:param kernel_size: size of the convolutional kernel (int or tuple)
		:param stride: stride of the convolutional kernel (int or tuple
		:param w_nrm: whether to normalize the weight vectors before computing outputs
		:param act: the nonlinear activation function after convolution
		:param mode: the learning mode, either 'swta' or 'hpca'
		:param k: softmax inverse temperature parameter used for swta-type learning
		:param patchwise: whether updates for each convolutional patch should be computed separately,
		and then aggregated
		:param contrast: coefficient that rescales negative compared to positive updates in contrastive-type learning
		:param uniformity: whether to use uniformity weighting in contrastive-type learning.
		:param alpha: weighting coefficient between hebbian and backprop updates (0 means fully backprop, 1 means fully hebbian).
		"""
        super(HebbianConv2d, self).__init__()
        # spatial competition is the appropriate comp modes
        self.mode = mode
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = padding
        self.padding_mode = 'reflect'
        if mode == "hard":
            self.padding_mode = 'symmetric'

        self.F_padding = (padding, padding, padding, padding)
        self.groups = 1  # in_channels for depthwise

        weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.abs(torch.randn((out_channels, in_channels // self.groups, *self.kernel_size))))

        # self.weight = nn.Parameter(torch.empty(out_channels, in_channels // self.groups, *self.kernel_size))
        # init.kaiming_uniform_(self.weight)

        # self.weight = center_surround_init(out_channels, in_channels, kernel_size, 1)
        print(self.weight.shape)

        self.w_nrm = w_nrm
        self.act = act
        # self.act = self.cos_sim2d
        self.theta_decay = 0.5
        if mode == "bcm":
            self.theta = nn.Parameter(torch.ones(out_channels), requires_grad=False)

        self.register_buffer('delta_w', torch.zeros_like(self.weight))
        self.top_k = top_k
        self.patchwise = patchwise
        self.contrast = contrast
        self.uniformity = uniformity
        self.alpha = alpha
        self.wta_competition = wta_competition
        self.lateral_inhibition_mode = lateral_competition
        self.lateral_learning_rate = lateral_inhibition_strength  # Adjust as needed
        self.lebesgue_p = 2

        self.prune_rate = prune_rate  # 99% of connections are pruned
        self.t_invert = torch.tensor(t_invert)

        self.presynaptic_competition_type = "softmax"
        self.presynaptic_weights = False  # presynaptic competition in forward pass

        self.activation_history = None
        self.temporal_window = 500
        self.competition_k = 2
        self.competition_type = "hard"

        if self.kernel !=1:
            self.sm_kernel = create_sm_kernel()
            self.register_buffer('surround_kernel', self.sm_kernel)
            self.visualize_surround_modulation_kernel()

    def visualize_surround_modulation_kernel(self):
        """
        Visualizes the surround modulation kernel using matplotlib.
        """
        sm_kernel = self.sm_kernel.squeeze().cpu().detach().numpy()
        plt.figure(figsize=(5, 5))
        plt.imshow(sm_kernel, cmap='jet')
        plt.colorbar()
        plt.title('Surround Modulation Kernel')
        plt.show()

    def apply_lebesgue_norm(self, w):
        return torch.sign(w) * torch.abs(w) ** (self.lebesgue_p - 1)

    def apply_weights(self, x, w):
        """
		This function provides the logic for combining input x and weight w
		"""
        # w = self.apply_lebesgue_norm(self.weight)
        # if self.padding != 0 and self.padding != None:
        # x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        x = symmetric_pad(x, self.padding)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def compute_activation(self, x):
        w = self.weight.abs()
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.presynaptic_weights: w = self.compute_presynaptic_competition_global(w)
        y = self.act(self.apply_weights(x, w))
        # Channel expansion with 1x1 conv
        # For cosine similarity activation if cosine is to be used for next layer
        # y = self.act(x)
        return y, w

    def forward(self, x):
        y, w = self.compute_activation(x)
        if self.kernel !=1:
            y = F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                     padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)
        if self.training:
            self.compute_update(x, y, w)
        return y

    def compute_update(self, x, y, weight):
        """
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
        if self.mode not in [self.MODE_HPCA, self.MODE_BASIC_HEBBIAN, self.MODE_WTA, self.MODE_BCM, self.MODE_SOFTWTA,
                             self.MODE_HARDWT, self.MODE_PRESYNAPTIC_COMPETITION, self.MODE_TEMPORAL_COMPETITION,
                             self.MODE_ADAPTIVE_THRESHOLD, self.MODE_ANTIHARDWT]:
            raise NotImplementedError(
                "Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))

        if self.mode == self.MODE_BASIC_HEBBIAN:
            # Compute yx using conv2d
            yx = F.conv2d(x.transpose(0, 1), y.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride)
            yx = yx.view(self.out_channels, self.in_channels, *self.kernel_size)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Reshape yx to match the weight shape
            yx = yx.view(weight.shape)
            # Compute y * w
            y_sum = y.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw = y_sum * weight
            # Compute update
            update = yx - yw
            # Normalization (optional, keeping it for consistency with original code)
            update.div_(torch.abs(update).amax() + 1e-30)
            self.delta_w += update

        if self.mode == self.MODE_HARDWT:
            batch_size, out_channels, height_out, width_out = y.shape
            y_flat = y.transpose(0, 1).reshape(out_channels, -1)
            win_neurons = torch.argmax(y_flat, dim=0)
            wta_mask = F.one_hot(win_neurons, num_classes=out_channels).float()
            wta_mask = wta_mask.transpose(0, 1).view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            y_wta = y * wta_mask
            # Compute yx using conv2d
            # Standard convolution
            yx = F.conv2d(x.transpose(0, 1), y_wta.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride).transpose(0, 1)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Compute yu
            yu = torch.sum(y_wta, dim=(0, 2, 3))
            # Compute update
            update = yx - yu.view(-1, 1, 1, 1) * weight
            # Normalization
            update.div_(torch.abs(update).amax() + 1e-30)
            self.delta_w += update

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
        new_weight = self.weight + 0.1 * self.alpha * self.delta_w
        # Ensure weights maintain their sign
        # Update weights
        self.weight.copy_(new_weight.abs())
        # Reset delta_w
        self.delta_w.zero_()
        # if self.weight.grad is None:
        #     self.weight.grad = -self.alpha * self.delta_w
        # else:
        #     self.weight.grad = (1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w
        # self.delta_w.zero_()
