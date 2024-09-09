import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torch.nn.init as init


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


def center_surround_init(out_channels, in_channels, kernel_size, groups=1):
    # Calculate weight range
    weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)

    # Calculate sigma based on kernel size (using equation 3 from the paper)
    gamma = torch.empty(out_channels).uniform_(0, 0.5)
    sigma = (kernel_size / 4) * torch.sqrt((1 - gamma ** 2) / (-torch.log(gamma)))

    # Create meshgrid for x and y coordinates
    x = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    y = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Calculate center and surround Gaussians
    center = torch.exp(-(xx ** 2 + yy ** 2) / (2 * (gamma.view(-1, 1, 1) * sigma.view(-1, 1, 1)) ** 2))
    surround = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma.view(-1, 1, 1) ** 2))

    # Calculate DoG (Difference of Gaussians)
    dog = center - surround

    # Normalize DoG
    ac = torch.sum(torch.clamp(dog, min=0))
    as_ = torch.sum(-torch.clamp(dog, max=0))
    dog = weight_range * 0.5 * dog / (ac + as_)

    # Assign excitatory (positive) or inhibitory (negative) centers
    center_type = torch.cat([torch.ones(out_channels // 2), -torch.ones(out_channels - out_channels // 2)])
    center_type = center_type[torch.randperm(out_channels)].view(-1, 1, 1)
    dog = dog * center_type

    # Repeat for in_channels and reshape to match conv2d weight shape
    dog = dog.unsqueeze(1).repeat(1, in_channels // groups, 1, 1)
    dog = dog.reshape(out_channels, in_channels // groups, kernel_size, kernel_size)

    return nn.Parameter(dog)

def create_sm_kernel(kernel_size=5, sigma_e=1.2, sigma_i=1.4):
    center = kernel_size // 2
    x, y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    x = x.float() - center
    y = y.float() - center

    gaussian_e = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_e ** 2))
    gaussian_i = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma_i ** 2))

    dog = gaussian_e / (2 * math.pi * sigma_e ** 2) - gaussian_i / (2 * math.pi * sigma_i ** 2)
    sm_kernel = dog / dog[center, center]

    return sm_kernel.unsqueeze(0).unsqueeze(0).to(device)

# Doubts:
# Visualizing weights, as separated between channels and spatial
#

class HebbianDepthConv2d(nn.Module):
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, groups = 1,
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
        super(HebbianDepthConv2d, self).__init__()
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
        self.groups = in_channels # in_channels for depthwise

        # # Depthwise separable weights
        weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.abs(torch.randn(in_channels, 1, *self.kernel_size)))

        # self.weight = nn.Parameter(torch.empty(in_channels, 1, *self.kernel_size))
        # init.kaiming_uniform_(self.weight)

        # self.weight = center_surround_init(in_channels, 1, kernel_size, 1)
        # print(self.weight.shape)

        self.w_nrm = w_nrm
        self.act = act
        # self.act = self.cos_sim2d
        self.theta_decay = 0.5
        if mode == "bcm":
            self.theta = nn.Parameter(torch.ones(out_channels))

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

        if kernel_size !=1:
            self.sm_kernel = create_sm_kernel()
            self.register_buffer('surround_kernel', self.sm_kernel)

    def apply_lebesgue_norm(self, w):
        return torch.sign(w) * torch.abs(w) ** (self.lebesgue_p - 1)

    def cosine(self, x, w):
        w_normalized = F.normalize(w, p=2, dim=1)
        conv_output = symmetric_pad(x, self.padding)
        conv_output = F.conv2d(conv_output, w_normalized, None, self.stride, 0, self.dilation, groups=self.groups)
        x_squared = x.pow(2)
        x_squared_sum = F.conv2d(x_squared, torch.ones_like(w), None, self.stride, self.padding, self.dilation,
                                 self.groups)
        x_norm = torch.sqrt(x_squared_sum + 1e-8)
        cosine_sim = conv_output / x_norm
        return cosine_sim

    def apply_weights(self, x, w):
        """
		This function provides the logic for combining input x and weight w
		"""
        # w = self.apply_lebesgue_norm(self.weight)
        # if self.padding != 0 and self.padding != None:
        # x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        x = symmetric_pad(x, self.padding)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def apply_surround_modulation(self, y):
        return F.conv2d(y, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                        padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)

    def compute_activation(self, x):
        w = self.weight.abs()
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.presynaptic_weights: w = self.compute_presynaptic_competition(w)
        # y_depthwise = self.act(self.apply_weights(x, w))
        # For cosine similarity activation if cosine is to be used for next layer
        y_depthwise = self.cosine(x,w)
        return y_depthwise, w

    def forward(self, x):
        y_depthwise, w = self.compute_activation(x)
        if self.kernel !=1:
            y_depthwise = self.apply_surround_modulation(y_depthwise)
        if self.training:
            self.compute_update(x, y_depthwise, w)
        return y_depthwise

    def compute_update(self, x, y, weight):
        if self.mode == self.MODE_BASIC_HEBBIAN:
            update = self.update_basic_hebbian(x, y, weight)
        elif self.mode == self.MODE_HARDWT:
            update = self.update_hardwt(x, y, weight)
        elif self.mode == self.MODE_SOFTWTA:
            update = self.update_softwta(x, y, weight)
        elif self.mode == self.MODE_ANTIHARDWT:
            update = self.update_antihardwt(x, y, weight)
        elif self.mode == self.MODE_BCM:
            update = self.update_bcm(x, y, weight)
        elif self.mode == self.MODE_TEMPORAL_COMPETITION:
            update = self.update_temporal_competition(x, y, weight)
        elif self.mode == self.MODE_ADAPTIVE_THRESHOLD:
            update = self.update_adaptive_threshold(x, y, weight)
        else:
            raise NotImplementedError(f"Learning mode {self.mode} unavailable for {self.__class__.__name__} layer")

        # Weight Normalization and added to weight change buffer
        update.div_(torch.abs(update).amax() + 1e-30)
        self.delta_w += update

    def update_basic_hebbian(self, x, y, weight):
        yx = self.compute_yx(x, y)
        y_sum = y.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        yw = y_sum * weight
        return yx - yw

    def update_hardwt(self, x, y, weight):
        y_wta = y * self.compute_wta_mask(y)
        yx = self.compute_yx(x, y_wta)
        yu = torch.sum(y_wta, dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        return yx - yu * weight

    def update_softwta(self, x, y, weight):
        softwta_activs = self.compute_softwta_activations(y)
        yx = self.compute_yx(x, softwta_activs)
        yu = torch.sum(torch.mul(softwta_activs, y), dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        return yx - yu * weight

    def update_bcm(self, x, y, weight):
        y_wta = y * self.compute_wta_mask(y)
        y_squared = y_wta.pow(2).mean(dim=(0, 2, 3))
        self.theta.data = (1 - self.theta_decay) * self.theta + self.theta_decay * y_squared
        y_minus_theta = y_wta - self.theta.view(1, -1, 1, 1)
        bcm_factor = y_wta * y_minus_theta
        yx = self.compute_yx(x, bcm_factor)
        return yx.view(weight.shape)

    def update_temporal_competition(self, x, y, weight):
        batch_size, out_channels, height_out, width_out = y.shape
        self.update_activation_history(y)
        temporal_winners = self.compute_temporal_winners(y)
        y_winners = temporal_winners * y
        y_winners = self.apply_competition(y_winners, batch_size, out_channels)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def update_adaptive_threshold(self, x, y, weight):
        batch_size, out_channels, height_out, width_out = y.shape
        similarities = F.conv2d(x, weight, stride=self.stride, padding=self.padding, groups=self.groups)
        similarities = similarities / (torch.norm(weight.view(out_channels, -1), dim=1).view(1, -1, 1, 1) + 1e-10)
        threshold = self.compute_adaptive_threshold(similarities)
        winners = (similarities > threshold).float()
        y_winners = winners * similarities
        y_winners = self.apply_competition(y_winners, batch_size, out_channels)
        yx = self.compute_yx(x, y_winners)
        y_sum = y_winners.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
        update = yx - y_sum * weight
        return update

    def compute_yx(self, x, y):
        yx = F.conv2d(x.transpose(0, 1), y.transpose(0, 1), padding=0,
                      stride=self.dilation, dilation=self.stride).transpose(0, 1)
        yx = yx.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)
        return yx

    def compute_wta_mask(self, y):
        batch_size, out_channels, height_out, width_out = y.shape
        # WTA competition within each channel
        y_flat = y.view(batch_size, out_channels, -1)
        win_neurons = torch.argmax(y_flat, dim=2)
        wta_mask = F.one_hot(win_neurons, num_classes=height_out * width_out).float()
        return wta_mask.view(batch_size, out_channels, height_out, width_out)

    def compute_softwta_activations(self, y):
        # Competition and anti-Hebbian learning for y_depthwise
        batch_size, in_channels, height_depthwise, width_depthwise = y.shape
        # Reshape to apply softmax within each channel
        y_depthwise_reshaped = y.view(batch_size, in_channels, -1)
        # Apply softmax within each channel
        flat_softwta_activs_depthwise = torch.softmax(self.t_invert * y_depthwise_reshaped, dim=2)
        # Turn all postsynaptic activations into anti-Hebbian
        flat_softwta_activs_depthwise = -flat_softwta_activs_depthwise
        # Find winners within each channel
        win_neurons_depthwise = torch.argmax(y_depthwise_reshaped, dim=2)
        # Create a mask to flip the sign of winning neurons
        mask = torch.zeros_like(flat_softwta_activs_depthwise)
        mask.scatter_(2, win_neurons_depthwise.unsqueeze(2), 1)
        # Flip the sign of winning neurons
        flat_softwta_activs_depthwise = flat_softwta_activs_depthwise * (1 - 2 * mask)
        # Reshape back to original shape
        return flat_softwta_activs_depthwise.view(batch_size, in_channels, height_depthwise,
                                                                      width_depthwise)

    def update_activation_history(self, y):
        if self.activation_history is None:
            self.activation_history = y.detach().clone()
        else:
            self.activation_history = torch.cat([self.activation_history, y.detach()], dim=0)
            if self.activation_history.size(0) > self.temporal_window:
                self.activation_history = self.activation_history[-self.temporal_window:]

    def compute_temporal_winners(self, y):
        batch_size, out_channels, height_out, width_out = y.shape
        history_spatial = self.activation_history.view(-1, out_channels, height_out, width_out)
        median_activations = torch.median(history_spatial, dim=0)[0]
        temporal_threshold = torch.mean(median_activations, dim=(1, 2), keepdim=True)
        return (median_activations > temporal_threshold).float()

    def compute_adaptive_threshold(self, similarities):
        mean_sim = similarities.mean(dim=(2, 3), keepdim=True)
        std_sim = similarities.std(dim=(2, 3), keepdim=True)
        return mean_sim + self.competition_k * std_sim

    def apply_competition(self, y, batch_size, out_channels):
        if self.mode in [self.MODE_TEMPORAL_COMPETITION, self.MODE_ADAPTIVE_THRESHOLD]:
            if self.competition_type == 'hard':
                y = y.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y, self.top_k, dim=2, largest=True, sorted=False).indices
                y_compete = torch.zeros_like(y)
                y_compete.scatter_(2, top_k_indices, y.gather(2, top_k_indices))
                return y_compete.view_as(y)
            elif self.competition_type == 'soft':
                return torch.softmax(self.t_invert * y.view(batch_size, out_channels, -1), dim=2).view_as(y)
        return y

    @torch.no_grad()
    def local_update(self):
        new_weight = self.weight + 0.1 * self.alpha * self.delta_w
        self.weight.copy_(new_weight.abs())
        self.delta_w.zero_()
