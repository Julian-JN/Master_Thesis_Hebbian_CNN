import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt


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
    MODE_BASIC_HEBBIAN = 'basic'
    MODE_SOFTWTA = 'soft'
    MODE_HARDWT = "hard"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, groups=1,
                 w_nrm=False, bias=False, act=nn.Identity(),
                 mode=MODE_HARDWT, k=1, patchwise=True,
                 contrast=1., uniformity=False, alpha=1., wta_competition='similarity_spatial',
                 lateral_competition="filter",
                 lateral_inhibition_strength=0.01, top_k=1, prune_rate=0, t_invert=1.):
        super(HebbianConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.padding = padding
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)

        self.groups = groups
        self.w_nrm = w_nrm
        self.act = act
        self.mode = mode
        self.alpha = alpha
        self.lr = 0.1
        self.presynaptic_weights = False

        # Define the number of excitatory and inhibitory neurons
        self.excitatory_channels = int(out_channels * 0.8)
        self.inhibitory_channels = out_channels - self.excitatory_channels

        # Initialize weights with the more efficient representation
        weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight_ee = nn.Parameter(weight_range * torch.abs(
            torch.randn(self.excitatory_channels, in_channels // groups, *self.kernel_size)))
        self.weight_ei = nn.Parameter(weight_range * torch.abs(
            torch.randn(self.inhibitory_channels, in_channels // groups, *self.kernel_size)))
        self.weight_ie = nn.Parameter(weight_range * torch.abs(
            torch.randn(self.excitatory_channels, self.inhibitory_channels, 1, 1)))

        self.weight = self.weight_ee

        # Initialize delta weights for updates
        self.register_buffer('delta_w_ee', torch.zeros_like(self.weight_ee))
        self.register_buffer('delta_w_ei', torch.zeros_like(self.weight_ei))
        self.register_buffer('delta_w_ie', torch.zeros_like(self.weight_ie))

        if self.kernel != 1:
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

    def forward(self, x):
        # Apply excitatory to excitatory weights
        y_ee = self.apply_weights(x, self.weight_ee)

        # Apply excitatory to inhibitory weights
        y_ei = self.apply_weights(x, self.weight_ei)

        # Apply inhibitory to excitatory weights
        y_ie = F.conv2d(y_ei, torch.abs(self.weight_ie), stride=1, padding=0)

        # Combine excitatory and inhibitory inputs
        y = F.relu(y_ee - y_ie)
        # Apply activation function
        y = self.act(y)

        if self.kernel != 1:
            y = F.conv2d(y, self.sm_kernel.repeat(self.excitatory_channels, 1, 1, 1),
                         padding=self.sm_kernel.size(-1) // 2, groups=self.excitatory_channels)

        if self.training:
            self.compute_update(x, y_ee, y_ei, y_ie, y)

        return y

    def apply_weights(self, x, w):
        x = symmetric_pad(x, self.padding)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def compute_update(self, x, y_ee, y_ei, y_ie, y):
        if self.mode == self.MODE_HARDWT:
            batch_size, _, height_out, width_out = y.shape

            # WTA for excitatory channels
            y_ee_flat = y_ee.transpose(0, 1).reshape(self.excitatory_channels, -1)
            win_neurons_ee = torch.argmax(y_ee_flat, dim=0)
            wta_mask_ee = F.one_hot(win_neurons_ee, num_classes=self.excitatory_channels).float()
            wta_mask_ee = wta_mask_ee.transpose(0, 1).view(self.excitatory_channels, batch_size, height_out,
                                                           width_out).transpose(0, 1)
            y_ee_wta = y_ee * wta_mask_ee

            # WTA for inhibitory channels
            y_ei_flat = y_ei.transpose(0, 1).reshape(self.inhibitory_channels, -1)
            win_neurons_ei = torch.argmax(torch.abs(y_ei_flat), dim=0)
            wta_mask_ei = F.one_hot(win_neurons_ei, num_classes=self.inhibitory_channels).float()
            wta_mask_ei = wta_mask_ei.transpose(0, 1).view(self.inhibitory_channels, batch_size, height_out,
                                                           width_out).transpose(0, 1)
            y_ei_wta = y_ei * wta_mask_ei

            # WTA for inhibited excitatory output
            y_ie_flat = y_ie.transpose(0, 1).reshape(self.excitatory_channels, -1)
            win_neurons_ie = torch.argmax(y_ie_flat, dim=0)
            wta_mask_ie = F.one_hot(win_neurons_ie, num_classes=self.excitatory_channels).float()
            wta_mask_ie = wta_mask_ie.transpose(0, 1).view(self.excitatory_channels, batch_size, height_out,
                                                           width_out).transpose(0, 1)
            y_ie_wta = y_ie * wta_mask_ie

            # Compute updates for E->E weights
            yx_ee = F.conv2d(x.transpose(0, 1), y_ee_wta.transpose(0, 1), padding=0,
                             stride=self.dilation, dilation=self.stride).transpose(0, 1)
            yu_ee = torch.sum(y_ee_wta, dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw_ee = yu_ee * self.weight_ee
            update_ee = yx_ee - yw_ee

            # Compute updates for E->I weights
            yx_ei = F.conv2d(x.transpose(0, 1), y_ei_wta.transpose(0, 1), padding=0,
                             stride=self.dilation, dilation=self.stride).transpose(0, 1)
            yu_ei = torch.sum(torch.abs(y_ei_wta), dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw_ei = yu_ei * self.weight_ei
            update_ei = yx_ei - yw_ei

            # Compute updates for I->E weights
            yx_ie = F.conv2d(y_ei_wta.transpose(0, 1), y_ie_wta.transpose(0, 1), padding=0,
                             stride=self.dilation, dilation=self.stride).transpose(0, 1)
            yu_ie = torch.sum(y_ie_wta, dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw_ie = yu_ie * self.weight_ie
            update_ie = yx_ie - yw_ie

            # Normalize updates
            update_ee.div_(torch.abs(update_ee).amax() + 1e-8)
            update_ei.div_(torch.abs(update_ei).amax() + 1e-8)
            update_ie.div_(torch.abs(update_ie).amax() + 1e-8)

            # Store updates
            self.delta_w_ee += update_ee
            self.delta_w_ei += update_ei
            self.delta_w_ie += update_ie

    @torch.no_grad()
    def local_update(self):
        # Apply update while respecting Dale's Law
        new_weight_ee = torch.abs(self.weight_ee + 0.1 * self.alpha * self.delta_w_ee)
        new_weight_ei = torch.abs(self.weight_ei + 0.1 * self.alpha * self.delta_w_ei)
        new_weight_ie = torch.abs(self.weight_ie + 0.1 * self.alpha * self.delta_w_ie)

        # Update weights
        self.weight_ee.copy_(new_weight_ee)
        self.weight_ei.copy_(new_weight_ei)
        self.weight_ie.copy_(new_weight_ie)

        # Reset delta weights
        self.delta_w_ee.zero_()
        self.delta_w_ei.zero_()
        self.delta_w_ie.zero_()
