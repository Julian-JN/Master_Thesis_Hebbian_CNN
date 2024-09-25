import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import matplotlib.pyplot as plt

import torch.nn.init as init

"""
    TODO:
    OPTIMIZE CODE:
        - Vectorization and In-Place operations
        - Optimize data movement between CPU and GPU
        - Simplify code: more modularity
        - Removed redundant computations: Some calculations that were repeated in different modes have been consolidated into separate methods.
        - Each learning mode now has its own update method, making it easier to maintain and extend.
        
    ABS Changed:
        - Unify Abs and Mixed together: Difference in weight intialisation, abs weights and update
        - Everything else the same?
"""

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
                 lateral_competition="combined",
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

        # weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        # self.weight = nn.Parameter(
        #     torch.rand((out_channels, in_channels // self.groups, *self.kernel_size)) * weight_range)

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

        if self.kernel != 1:
            self.sm_kernel = create_sm_kernel()
            self.register_buffer('surround_kernel', self.sm_kernel)
            self.visualize_surround_modulation_kernel()

        self.target_activity = 0.1
        self.scaling_rate = 0.001
        self.register_buffer('average_activity', torch.zeros(out_channels))

        self.growth_probability = 0.1
        self.new_synapse_strength = 1
        self.prune_threshold_percentile = 10  # Prune bottom 10% of weights

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
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def update_average_activity(self, y):
        current_activity = y.mean(dim=(0, 2, 3))
        self.average_activity = 0.9 * self.average_activity + 0.1 * current_activity

    def synaptic_scaling(self):
        scale_factor = self.target_activity / (self.average_activity + 1e-6)
        self.weight.data *= (1 + self.scaling_rate * (scale_factor - 1)).view(-1, 1, 1, 1)

    def structural_plasticity(self):
        with torch.no_grad():
            # Pruning step
            prune_threshold = torch.quantile(torch.abs(self.weight), self.prune_threshold_percentile / 100)
            weak_synapses = torch.abs(self.weight) < prune_threshold
            self.weight.data[weak_synapses] = 0
            # Growth step
            zero_weights = self.weight.data == 0
            new_synapses = torch.rand_like(self.weight) < self.growth_probability
            new_synapses &= zero_weights
            self.weight.data[new_synapses] = torch.randn_like(self.weight)[new_synapses] * self.new_synapse_strength

    def cosine(self, x, w):
        w_normalized = F.normalize(w, p=2, dim=1)
        # conv_output = symmetric_pad(x, self.padding)
        conv_output = F.conv2d(x, w_normalized, None, self.stride, 0, self.dilation, groups=self.groups)
        x_squared = x.pow(2)
        x_squared_sum = F.conv2d(x_squared, torch.ones_like(w), None, self.stride, 0, self.dilation,
                                 self.groups)
        x_norm = torch.sqrt(x_squared_sum + 1e-8)
        cosine_sim = conv_output / x_norm
        return cosine_sim

    def apply_surround_modulation(self, y):
        return F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                        padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)

    def compute_activation(self, x):
        x = symmetric_pad(x, self.padding)
        w = self.weight.abs()
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.presynaptic_weights: w = self.compute_presynaptic_competition_global(w)
        # y = self.act(self.apply_weights(x, w))
        # For cosine similarity activation if cosine is to be used for next layer
        y = self.cosine(x, w)
        return y, w

    def forward(self, x):
        y, w = self.compute_activation(x)
        # if self.lateral_inhibition_mode == "combined":
        #     y = self.combined_lateral_inhibition(y)
        if self.kernel != 1:
            y = self.apply_surround_modulation(y)
        if self.training:
            # self.update_average_activity(y)
            # self.synaptic_scaling()
            self.compute_update(x, y, w)
        return y

    def compute_update(self, x, y, weight):
        if self.mode == self.MODE_BASIC_HEBBIAN:
            update = self.update_basic_hebbian(x, y, weight)
        elif self.mode == self.MODE_HARDWT:
            update = self.update_hardwt(x, y, weight)
        elif self.mode == self.MODE_SOFTWTA:
            update = self.update_softwta(x, y, weight)
        elif self.mode == self.MODE_BCM:
            update = self.update_bcm(x, y, weight)
        else:
            raise NotImplementedError(f"Learning mode {self.mode} unavailable for {self.__class__.__name__} layer")
        # Weight Normalization and added to weight change buffer
        update.div_(torch.abs(update).amax() + 1e-30)
        self.delta_w += update

    def update_basic_hebbian(self, x, y, weight):
        # Grossberg Instar rule
        yx = self.compute_yx(x, y)
        y_sum = y.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
        yw = y_sum * weight
        update = yx - yw
        return update

    def update_hardwt(self, x, y, weight):
        # Grossberg Instar with wta mask
        y_wta = y * self.compute_wta_mask(y)
        yx = self.compute_yx(x, y_wta)
        yu = torch.sum(y_wta, dim=(0, 2, 3))
        update = yx - yu.view(-1, 1, 1, 1) * weight
        return update

    def update_softwta(self, x, y, weight):
        # SoftHebb Grossberg Instar Variation
        softwta_activs = self.compute_softwta_activations(y)
        yx = self.compute_yx(x, softwta_activs)
        yu = torch.sum(torch.mul(softwta_activs, y), dim=(0, 2, 3))
        update = yx - yu.view(-1, 1, 1, 1) * weight
        return update

    def update_bcm(self, x, y, weight):
        # BCM learning rule with WTA mask (remove if not needed)
        y_wta = y * self.compute_wta_mask(y)
        y_squared = y_wta.pow(2).mean(dim=(0, 2, 3))
        self.theta.data = (1 - self.theta_decay) * self.theta + self.theta_decay * y_squared
        y_minus_theta = y_wta - self.theta.view(1, -1, 1, 1)
        bcm_factor = y_wta * y_minus_theta
        yx = self.compute_yx(x, bcm_factor)
        update = yx.view(weight.shape)
        return update

    def compute_yx(self, x, y):
        # Computes common y*w term from Grossberg Instar
        yx = F.conv2d(x.transpose(0, 1), y.transpose(0, 1), padding=0,
                      stride=self.dilation, dilation=self.stride).transpose(0, 1)
        if self.groups != 1:
            yx = yx.mean(dim=1, keepdim=True)
        return yx

    def compute_wta_mask(self, y):
        # Computes WTA Mask
        batch_size, out_channels, height_out, width_out = y.shape
        y_flat = y.transpose(0, 1).reshape(out_channels, -1)
        win_neurons = torch.argmax(y_flat, dim=0)
        wta_mask = F.one_hot(win_neurons, num_classes=out_channels).float()
        return wta_mask.transpose(0, 1).view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

    def compute_softwta_activations(self, y):
        # Computes SoftHebb mask and Hebb/AntiHebb implementation
        batch_size, out_channels, height_out, width_out = y.shape
        flat_weighted_inputs = y.transpose(0, 1).reshape(out_channels, -1)
        flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
        flat_softwta_activs = -flat_softwta_activs
        win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
        competing_idx = torch.arange(flat_weighted_inputs.size(1))
        flat_softwta_activs[win_neurons, competing_idx] = -flat_softwta_activs[win_neurons, competing_idx]
        return flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

    def compute_antihardwta_activations(self, y):
        # Computes Hard-WTA mask and Hebb/AntiHebb implementation
        batch_size, out_channels, height_out, width_out = y.shape
        flat_weighted_inputs = y.transpose(0, 1).reshape(out_channels, -1)
        win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
        competing_idx = torch.arange(flat_weighted_inputs.size(1))
        anti_hebbian_mask = torch.ones_like(flat_weighted_inputs) * -1
        anti_hebbian_mask[win_neurons, competing_idx] = 1
        flat_hardwta_activs = flat_weighted_inputs * anti_hebbian_mask
        return flat_hardwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)

    @torch.no_grad()
    # Weight Update
    def local_update(self):
        new_weight = self.weight + 0.1 * self.alpha * self.delta_w
        # Update weights
        self.weight.copy_(new_weight.abs())
        # self.structural_plasticity()
        self.delta_w.zero_()
