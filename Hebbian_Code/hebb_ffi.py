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
    if padding == 0:
        return x
    return F.pad(x, (padding,) * 4, mode='reflect')


def create_sm_kernel(kernel_size=5, sigma_e=1.2, sigma_i=1.4):
    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing="ij")
    x = x.float() - center
    y = y.float() - center
    gaussian_e = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_e ** 2))
    gaussian_i = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_i ** 2))
    dog = gaussian_e / (2 * math.pi * sigma_e ** 2) - gaussian_i / (2 * math.pi * sigma_i ** 2)
    sm_kernel = dog / dog[center, center]
    return sm_kernel.unsqueeze(0).unsqueeze(0).to(device)


class HebbianConv2d(nn.Module):
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

        self.excitatory_channels = int(out_channels * 0.8)
        self.inhibitory_channels = out_channels - self.excitatory_channels

        # Initialize weights
        weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.w_ee = nn.Parameter(
            weight_range * torch.abs(torch.randn((self.excitatory_channels, in_channels // groups, *self.kernel_size))))
        self.w_ei = nn.Parameter(weight_range * torch.abs(
            torch.randn((self.inhibitory_channels, self.excitatory_channels // groups, *self.kernel_size))))
        self.w_ie = nn.Parameter(-weight_range * torch.abs(
            torch.randn((self.excitatory_channels, self.inhibitory_channels // groups, *self.kernel_size))))

        # Initialize delta weights
        self.register_buffer('delta_w_ee', torch.zeros_like(self.w_ee))
        self.register_buffer('delta_w_ei', torch.zeros_like(self.w_ei))
        self.register_buffer('delta_w_ie', torch.zeros_like(self.w_ie))

        if self.kernel != 1:
            self.sm_kernel = create_sm_kernel()
            self.register_buffer('surround_kernel', self.sm_kernel)

        # Initialize feedforward inhibition strength
        self.ff_inhibition_strength = nn.Parameter(torch.ones(self.inhibitory_channels, 1, 1, 1) * 0.1)

    def forward(self, x):
        # Excitatory to excitatory connections
        y_e = self.apply_weights(x, self.w_ee)
        y_e = self.act(y_e)

        # Excitatory to inhibitory connections
        y_i = self.apply_weights(y_e, self.w_ei)
        y_i = F.relu(y_i)  # Ensure non-negative activation for inhibitory neurons

        # Inhibitory to excitatory connections (feedforward inhibition)
        y_ie = self.apply_weights(y_i, self.w_ie)

        # Combine excitatory and inhibitory influences
        y = y_e + y_ie

        if self.kernel != 1:
            y = F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                         padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)

        if self.training:
            self.compute_update(x, y_e, y_i, y)

        return y

    def apply_weights(self, x, w):
        if self.w_nrm:
            w = normalize(w, dim=(1, 2, 3))
        x = symmetric_pad(x, self.padding)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def compute_update(self, x, y_e, y_i, y):
        if self.mode == self.MODE_HARDWT:
            batch_size, _, height_out, width_out = y.shape

            # WTA for excitatory channels
            y_exc_flat = y_e.transpose(0, 1).reshape(self.excitatory_channels, -1)
            win_neurons_exc = torch.argmax(y_exc_flat, dim=0)
            wta_mask_exc = F.one_hot(win_neurons_exc, num_classes=self.excitatory_channels).float()
            wta_mask_exc = wta_mask_exc.transpose(0, 1).view(self.excitatory_channels, batch_size, height_out,
                                                             width_out).transpose(0, 1)
            y_wta_exc = y_e * wta_mask_exc

            # WTA for inhibitory channels
            y_inh_flat = y_i.transpose(0, 1).reshape(self.inhibitory_channels, -1)
            win_neurons_inh = torch.argmax(torch.abs(y_inh_flat), dim=0)
            wta_mask_inh = F.one_hot(win_neurons_inh, num_classes=self.inhibitory_channels).float()
            wta_mask_inh = wta_mask_inh.transpose(0, 1).view(self.inhibitory_channels, batch_size, height_out,
                                                             width_out).transpose(0, 1)
            y_wta_inh = y_i * wta_mask_inh

            # Compute updates
            yx_ee = F.conv2d(x.transpose(0, 1), y_wta_exc.transpose(0, 1), padding=0, stride=self.dilation,
                             dilation=self.stride).transpose(0, 1)
            yx_ei = F.conv2d(y_wta_exc.transpose(0, 1), y_wta_inh.transpose(0, 1), padding=0, stride=self.dilation,
                             dilation=self.stride).transpose(0, 1)
            yx_ie = F.conv2d(y_wta_inh.transpose(0, 1), y.transpose(0, 1), padding=0, stride=self.dilation,
                             dilation=self.stride).transpose(0, 1)

            yu_exc = torch.sum(y_wta_exc, dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yu_inh = torch.sum(torch.abs(y_wta_inh), dim=(0, 2, 3)).view(-1, 1, 1, 1)

            yw_ee = yu_exc * self.w_ee
            yw_ei = yu_inh * self.w_ei
            yw_ie = yu_exc * self.w_ie

            update_ee = yx_ee - yw_ee
            update_ei = yx_ei - yw_ei
            update_ie = yx_ie - yw_ie

            update_ee.div_(torch.abs(update_ee).amax() + 1e-8)
            update_ei.div_(torch.abs(update_ei).amax() + 1e-8)
            update_ie.div_(torch.abs(update_ie).amax() + 1e-8)

            self.delta_w_ee += update_ee
            self.delta_w_ei += update_ei
            self.delta_w_ie += update_ie

    @torch.no_grad()
    def local_update(self):
        # Apply updates while respecting Dale's Law
        new_w_ee = torch.abs(self.w_ee + 0.1 * self.alpha * self.delta_w_ee)
        new_w_ei = torch.abs(self.w_ei + 0.1 * self.alpha * self.delta_w_ei)
        new_w_ie = -torch.abs(self.w_ie + 0.1 * self.alpha * self.delta_w_ie)

        # Update weights
        self.w_ee.copy_(new_w_ee)
        self.w_ei.copy_(new_w_ei)
        self.w_ie.copy_(new_w_ie)

        # Reset delta weights
        self.delta_w_ee.zero_()
        self.delta_w_ei.zero_()
        self.delta_w_ie.zero_()

    def visualize_surround_modulation_kernel(self):
        if hasattr(self, 'sm_kernel'):
            sm_kernel = self.sm_kernel.squeeze().cpu().detach().numpy()
            plt.figure(figsize=(5, 5))
            plt.imshow(sm_kernel, cmap='jet')
            plt.colorbar()
            plt.title('Surround Modulation Kernel')
            plt.show()
        else:
            print("Surround modulation kernel is not available for this layer.")