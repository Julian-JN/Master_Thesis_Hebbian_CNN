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
        self.weight = nn.Parameter(weight_range * torch.randn(in_channels, 1, *self.kernel_size))

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
        w = self.weight
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.presynaptic_weights: w = self.compute_presynaptic_competition(w)
        y_depthwise = self.act(self.apply_weights(x, w))
        # Channel expansion with 1x1 conv
        # For cosine similarity activation if cosine is to be used for next layer
        # y = self.act(x)
        return y_depthwise, w

    def forward(self, x):
        y_depthwise, w = self.compute_activation(x)
        if self.kernel !=1:
            y_depthwise = F.conv2d(y_depthwise, self.sm_kernel.repeat(self.out_channels, 1, 1, 1),
                     padding=self.sm_kernel.size(-1) // 2, groups=self.out_channels)
        if self.training:
            self.compute_update(x, y_depthwise, w)
        return y_depthwise

    def compute_update(self, x, y_depthwise, weight):
        """
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
        if self.mode not in [self.MODE_HPCA, self.MODE_BASIC_HEBBIAN, self.MODE_WTA, self.MODE_BCM, self.MODE_SOFTWTA,
                             self.MODE_HARDWT, self.MODE_PRESYNAPTIC_COMPETITION, self.MODE_TEMPORAL_COMPETITION, self.MODE_ADAPTIVE_THRESHOLD, self.MODE_ANTIHARDWT]:
            raise NotImplementedError(
                "Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))

        if self.mode == self.MODE_BASIC_HEBBIAN:
            # Update for depthwise convolution
            yx = F.conv2d(x.transpose(0, 1), y_depthwise.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride)
            # yx shape: [in_channels, in_channels, kernel_height, kernel_width]
            # We need to keep only the diagonal elements for each channel
            yx = yx.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)
            # yx shape after diagonal: [in_channels, 1, kernel_height, kernel_width]
            y_depthwise_sum = y_depthwise.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
            yw = y_depthwise_sum * weight
            update_depthwise = yx - yw
            update_depthwise.div_(torch.abs(update_depthwise).amax() + 1e-30)
            self.delta_w += update_depthwise

        if self.mode == self.MODE_PRESYNAPTIC_COMPETITION:
            # Compute yx using conv2d with input x
            yx = F.conv2d(x.transpose(0, 1), y_depthwise.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride)
            yx = yx.view(self.out_channels, self.in_channels, *self.kernel_size)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Reshape yx to match the weight shape
            yx = yx.view(weight.shape)
            # Apply competition to yx
            yx_competed = yx * weight
            # Compute y * w
            y_sum = y_depthwise.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw = y_sum * weight
            # Compute update
            update = yx_competed - yw
            # Normalization
            update.div_(torch.abs(update).amax() + 1e-30)
            self.delta_w += update

        if self.mode == self.MODE_TEMPORAL_COMPETITION:
            batch_size, out_channels, height_out, width_out = y_depthwise.shape
            # Update activation history
            if self.activation_history is None:
                self.activation_history = y_depthwise.detach().clone()
            else:
                self.activation_history = torch.cat([self.activation_history, y_depthwise.detach()], dim=0)
                if self.activation_history.size(0) > self.temporal_window:
                    self.activation_history = self.activation_history[-self.temporal_window:]
            # Reshape activation history to group by spatial location
            history_spatial = self.activation_history.view(-1, self.out_channels, height_out, width_out)
            # Compute median activations for each spatial location
            median_activations = torch.median(history_spatial, dim=0)[0]
            # Compute threshold for each spatial location
            temporal_threshold = torch.mean(median_activations, dim=0, keepdim=True)
            # Determine winners at each spatial location
            temporal_winners = (median_activations > temporal_threshold).float()
            y_winners = temporal_winners * y_depthwise
            if self.competition_type == 'hard':
                y_winners = y_winners.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y_winners, self.top_k, dim=1, largest=True, sorted=False).indices
                y_compete = torch.zeros_like(y_winners)
                y_compete.scatter_(1, top_k_indices, y_winners.gather(1, top_k_indices))
                y_winners = y_compete.view_as(y_depthwise)
            elif self.competition_type == 'soft':
                y_winners = torch.softmax(self.t_invert * y_winners.view(batch_size, out_channels, -1), dim=1).view_as(
                    y_depthwise)
            elif self.competition_type == 'anti':
                y_winners = y_winners.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y_winners, self.top_k, dim=1, largest=True, sorted=False).indices
                anti_hebbian_mask = torch.ones_like(y_winners)
                anti_hebbian_mask.scatter_(1, top_k_indices, -1)
                y_winners = (y_winners * anti_hebbian_mask).view_as(y_depthwise)
            # Shape: [batch_size, out_channels, height_out, width_out]
            # Compute update using conv2d and conv_transpose2d
            yx = F.conv2d(x.transpose(0, 1), y_winners.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride)
            yx = yx.view(self.out_channels, self.in_channels, *self.kernel_size)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Reshape yx to match the weight shape
            yx = yx.view(weight.shape)
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            y_sum = y_winners.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
            # Shape: [out_channels, 1, 1, 1]
            update = yx - y_sum * weight
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            # Normalize update
            update = update / (torch.norm(update.view(out_channels, -1), dim=1).view(-1, 1, 1, 1) + 1e-10)
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            self.delta_w += update

        if self.mode == self.MODE_ADAPTIVE_THRESHOLD:
            batch_size, out_channels, height_out, width_out = y_depthwise.shape
            # Compute similarities using conv2d for efficiency
            similarities = F.conv2d(x, weight, stride=self.stride, padding=self.padding)
            # Shape: [batch_size, out_channels, height_out, width_out]
            similarities = similarities / (torch.norm(weight.view(out_channels, -1), dim=1).view(1, -1, 1, 1) + 1e-10)
            # Shape: [batch_size, out_channels, height_out, width_out]
            # Compute mean and std dev for each spatial location
            mean_sim = similarities.mean(dim=1, keepdim=True)
            std_sim = similarities.std(dim=1, keepdim=True)
            # Compute threshold for each spatial location
            threshold = mean_sim + self.competition_k * std_sim
            # Determine winners at each spatial location
            winners = (similarities > threshold).float()
            y_winners = winners * similarities # instead if similarity
            # Shape: [batch_size, out_channels, height_out, width_out]
            if self.competition_type == 'hard':
                y_winners = y_winners.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y_winners, self.top_k, dim=1, largest=True, sorted=False).indices
                y_compete = torch.zeros_like(y_winners)
                y_compete.scatter_(1, top_k_indices, y_winners.gather(1, top_k_indices))
                y_winners = y_compete.view_as(y_depthwise)
            elif self.competition_type == 'soft':
                y_winners = torch.softmax(self.t_invert * y_winners.view(batch_size, out_channels, -1), dim=1).view_as(
                    y_depthwise)
            elif self.competition_type == 'anti':
                y_winners = y_winners.view(batch_size, out_channels, -1)
                top_k_indices = torch.topk(y_winners, self.top_k, dim=1, largest=True, sorted=False).indices
                anti_hebbian_mask = torch.ones_like(y_winners)
                anti_hebbian_mask.scatter_(1, top_k_indices, -1)
                y_winners = (y_winners * anti_hebbian_mask).view_as(y_depthwise)
            # Compute update using conv2d
            yx = F.conv2d(x.transpose(0, 1), y_winners.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride)
            yx = yx.view(self.out_channels, self.in_channels, *self.kernel_size)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Reshape yx to match the weight shape
            yx = yx.view(weight.shape)
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            y_sum = y_winners.sum(dim=(0, 2, 3)).view(-1, 1, 1, 1)
            # Shape: [out_channels, 1, 1, 1]
            update = yx - y_sum * weight
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            # Normalize update
            update = update / (torch.norm(update.view(out_channels, -1), dim=1).view(-1, 1, 1, 1) + 1e-10)
            # Shape: [out_channels, in_channels, kernel_height, kernel_width]
            self.delta_w += update

        if self.mode == self.MODE_BCM:
            # batch_size, out_channels, height_out, width_out = y.shape
            # # Compute soft WTA using softmax (identical to SOFTWTA mode)
            # flat_weighted_inputs = y.transpose(0, 1).reshape(out_channels, -1)
            # flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            # flat_softwta_activs = -flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            #
            # # Find winning neurons
            # win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
            # competing_idx = torch.arange(flat_weighted_inputs.size(1))
            #
            # # Turn winner neurons' activations back to hebbian
            # flat_softwta_activs[win_neurons, competing_idx] = -flat_softwta_activs[win_neurons, competing_idx]
            #
            # # Reshape softwta activations
            # y_soft = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # # Update theta (sliding threshold) using WTA output
            # y_squared = y_soft.pow(2).mean(dim=(0, 2, 3))
            # self.theta.data = (1 - self.theta_decay) * self.theta + self.theta_decay * y_squared
            # # Compute BCM update with WTA
            # y_minus_theta = y_soft - self.theta.view(1, -1, 1, 1)
            # bcm_factor = y_soft * y_minus_theta
            # # Compute update using conv2d for consistency with original code
            # yx = F.conv2d(x.transpose(0, 1), bcm_factor.transpose(0, 1), padding=0,
            #               stride=self.dilation, dilation=self.stride).transpose(0, 1)
            # yx = yx.view(self.out_channels, self.in_channels, *self.kernel_size)
            # if self.groups != 1:
            #     yx = yx.mean(dim=1, keepdim=True)
            # # Reshape yx to match the weight shape
            # yx = yx.view(weight.shape)
            # # Compute update
            # update = yx.view(weight.shape)
            # # Normalize update (optional, keeping it for consistency with original code)
            # update.div_(torch.abs(update).amax() + 1e-30)
            # self.delta_w += update

            # # BCM using WT Competition
            batch_size, out_channels, height_out, width_out = y_depthwise.shape
            # WTA competition
            y_flat = y_depthwise.transpose(0, 1).reshape(out_channels, -1)
            win_neurons = torch.argmax(y_flat, dim=0)
            wta_mask = F.one_hot(win_neurons, num_classes=out_channels).float()
            wta_mask = wta_mask.transpose(0, 1).view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            y_wta = y_depthwise * wta_mask
            # Update theta (sliding threshold) using WTA output
            y_squared = y_wta.pow(2).mean(dim=(0, 2, 3))
            self.theta.data = (1 - self.theta_decay) * self.theta + self.theta_decay * y_squared
            # Compute BCM update with WTA
            y_minus_theta = y_wta - self.theta.view(1, -1, 1, 1)
            bcm_factor = y_wta * y_minus_theta
            # Compute update using conv2d for consistency with original code
            yx = F.conv2d(x.transpose(0, 1), bcm_factor.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride, groups=1).transpose(0, 1)
            if self.groups != 1:
                yx = yx.mean(dim=1, keepdim=True)
            # Compute update
            update = yx.view(weight.shape)
            # Normalize update (optional, keeping it for consistency with original code)
            update.div_(torch.abs(update).amax() + 1e-30)
            self.delta_w += update

        elif self.mode == self.MODE_SOFTWTA:
            # Competition and anti-Hebbian learning for y_depthwise
            batch_size, in_channels, height_depthwise, width_depthwise = y_depthwise.shape
            # Reshape to apply softmax within each channel
            y_depthwise_reshaped = y_depthwise.view(batch_size, in_channels, -1)

            # # Oscillations and Synchrony
            # if not hasattr(self, 'oscillation_phase'):
            #     self.oscillation_phase = 0
            # oscillation = 0.1 * torch.sin(torch.tensor(self.oscillation_phase))
            # y_depthwise_reshaped += oscillation
            # self.oscillation_phase += 0.1  # Update phase

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
            softwta_activs_depthwise = flat_softwta_activs_depthwise.view(batch_size, in_channels, height_depthwise,
                                                                          width_depthwise)
            # Update for depthwise convolution
            yx_depthwise = F.conv2d(x.transpose(0, 1), softwta_activs_depthwise.transpose(0, 1), padding=0,
                                    stride=self.dilation, dilation=self.stride)
            yx_depthwise = yx_depthwise.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)
            yu_depthwise = torch.sum(torch.mul(softwta_activs_depthwise, y_depthwise), dim=(0, 2, 3)).view(
                self.in_channels, 1, 1, 1)
            update_depthwise = yx_depthwise - yu_depthwise * weight
            update_depthwise.div_(torch.abs(update_depthwise).amax() + 1e-30)
            self.delta_w += update_depthwise

        if self.mode == self.MODE_ANTIHARDWT:
            batch_size, out_channels, height_out, width_out = y_depthwise.shape
            # Reshape y for easier processing
            flat_weighted_inputs = y_depthwise.transpose(0, 1).reshape(out_channels, -1)
            # Find winning neurons
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)
            competing_idx = torch.arange(flat_weighted_inputs.size(1))
            # Create anti-Hebbian mask
            anti_hebbian_mask = torch.ones_like(flat_weighted_inputs) * -1  # All neurons start as anti-Hebbian
            # Set winning neurons to Hebbian (positive)
            anti_hebbian_mask[win_neurons, competing_idx] = 1
            # Apply the mask to y
            flat_hardwta_activs = flat_weighted_inputs * anti_hebbian_mask
            # Reshape hardwta activations
            hardwta_activs = flat_hardwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # Compute yx using conv2d
            yx = F.conv2d(x.transpose(0, 1), hardwta_activs.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride).transpose(0, 1)
            if self.groups !=1:
                yx = yx.mean(dim=1, keepdim=True)
            # Compute yu
            yu = torch.sum(torch.mul(hardwta_activs, y_depthwise), dim=(0, 2, 3))
            # Compute update
            update = yx - yu.view(-1, 1, 1, 1) * weight
            # Normalization
            update.div_(torch.abs(update).amax() + 1e-30)
            self.delta_w += update

        if self.mode == self.MODE_HARDWT:
            # Competition for y_depthwise
            batch_size, in_channels, height_depthwise, width_depthwise = y_depthwise.shape
            y_depthwise_flat = y_depthwise.view(batch_size, in_channels, -1)
            win_neurons_depthwise = torch.argmax(y_depthwise_flat, dim=2)
            wta_mask_depthwise = F.one_hot(win_neurons_depthwise,
                                           num_classes=height_depthwise * width_depthwise).float()
            wta_mask_depthwise = wta_mask_depthwise.view(batch_size, in_channels, height_depthwise, width_depthwise)
            y_depthwise_wta = y_depthwise * wta_mask_depthwise
            # y_depthwise_wta = y_depthwise

            # Update for depthwise convolution
            yx_depthwise = F.conv2d(x.transpose(0, 1), y_depthwise_wta.transpose(0, 1), padding=0,
                                    stride=self.dilation, dilation=self.stride)
            yx_depthwise = yx_depthwise.diagonal(dim1=0, dim2=1).permute(2, 0, 1).unsqueeze(1)
            y_depthwise_sum = y_depthwise_wta.sum(dim=(0, 2, 3)).view(self.in_channels, 1, 1, 1)
            yw_depthwise = y_depthwise_sum * weight
            update_depthwise = yx_depthwise - yw_depthwise
            update_depthwise.div_(torch.abs(update_depthwise).amax() + 1e-30)
            self.delta_w += update_depthwise

    def generate_mask(self):
        return torch.bernoulli(torch.full_like(self.weight, 1 - self.prune_rate))

    def compute_presynaptic_competition(self, m):
        m = 1 / (torch.abs(m) + 1e-6)
        if self.presynaptic_competition_type == 'linear':
            return m / (m.sum(dim=0, keepdim=True) + 1e-6)
        elif self.presynaptic_competition_type == 'softmax':
            return F.softmax(m, dim=0)
        elif self.presynaptic_competition_type == 'lp_norm':
            return F.normalize(m, p=2, dim=0)
        else:
            raise ValueError(f"Unknown competition type: {self.competition_type}")

    def cos_sim2d(self, x):
        # Unfold the input
        w = self.weight
        x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        # x_unf shape: [batch_size, C*k*k, H*W]
        batch_size, channels_x_kernel, hw = x_unf.shape
        out_channels, _, kernel_h, kernel_w = w.shape
        # Reshape weights
        w_reshaped = w.view(out_channels, -1)  # Shape: [out_channels, C*k*k]
        # Reshape x_unf for batch matrix multiplication
        x_unf = x_unf.transpose(1, 2).reshape(batch_size * hw, channels_x_kernel)
        # Compute dot product
        dot_product = torch.matmul(x_unf, w_reshaped.t())  # Shape: [batch_size*H*W, out_channels]
        # Compute norms
        norm_x = torch.norm(x_unf, p=2, dim=1, keepdim=True)  # Shape: [batch_size*H*W, 1]
        norm_w = torch.norm(w_reshaped, p=2, dim=1, keepdim=True)  # Shape: [out_channels, 1]
        # Avoid division by zero
        norm_x[norm_x == 0] = 1e-8
        norm_w[norm_w == 0] = 1e-8
        # Compute cosine similarity
        cosine_similarity = dot_product / (norm_x * norm_w.t())
        # Reshape to match the original output shape
        out_shape = (batch_size,
                     (x.size(2) - self.kernel_size[0]) // self.stride[0] + 1,
                     (x.size(3) - self.kernel_size[1]) // self.stride[1] + 1,
                     out_channels)
        cosine_similarity = cosine_similarity.view(*out_shape)
        # Permute to get [batch_size, out_channels, H, W]
        cosine_similarity = cosine_similarity.permute(0, 3, 1, 2)
        return cosine_similarity

    def lateral_inhibition_within_filter(self, y_normalized, out_channels, hw):
        # y_normalized shape is [hw, out_channels] = [50176, 96]
        y_reshaped = y_normalized.t()  # Shape: [96, 50176]
        # Compute coactivation within each filter
        coactivation = torch.mm(y_reshaped, y_reshaped.t())  # Shape: [96, 96]
        # Update lateral weights using Anti-Hebbian learning
        if not hasattr(self, 'lateral_weights_filter'):
            self.lateral_weights_filter = torch.zeros_like(coactivation)
        lateral_update = self.lateral_learning_rate * (coactivation - self.lateral_weights_filter)
        self.lateral_weights_filter -= lateral_update  # Anti-Hebbian update
        # Apply inhibition
        inhibition = torch.mm(self.lateral_weights_filter, y_reshaped)
        y_inhibited = F.relu(y_reshaped - inhibition)
        return y_inhibited.t()  # Shape: [50176, 96]

    def lateral_inhibition_same_patch(self, y_normalized, out_channels, hw):
        # y_normalized shape is [hw, out_channels] = [50176, 96]
        y_reshaped = y_normalized
        # Compute coactivation for neurons looking at the same patch
        coactivation = torch.mm(y_reshaped.t(), y_reshaped)  # Shape: [96, 96]
        # Update lateral weights using Anti-Hebbian learning
        if not hasattr(self, 'lateral_weights_patch'):
            self.lateral_weights_patch = torch.zeros_like(coactivation)
        lateral_update = self.lateral_learning_rate * (coactivation - self.lateral_weights_patch)
        self.lateral_weights_patch -= lateral_update  # Anti-Hebbian update
        # Apply inhibition
        inhibition = torch.mm(y_reshaped, self.lateral_weights_patch)
        y_inhibited = F.relu(y_reshaped - inhibition)
        return y_inhibited  # Shape: [50176, 96]

    def combined_lateral_inhibition(self, y_normalized, out_channels, hw):
        # Apply inhibition within filter
        y_inhibited_filter = self.lateral_inhibition_within_filter(y_normalized, out_channels, hw)
        # Apply inhibition for the same patch
        y_inhibited_patch = self.lateral_inhibition_same_patch(y_normalized, out_channels, hw)
        # Combine the inhibitions (you can adjust the weighting as needed)
        y_inhibited = 0.5 * y_inhibited_filter + 0.5 * y_inhibited_patch
        return y_inhibited  # Shape: [50176, 96]

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
