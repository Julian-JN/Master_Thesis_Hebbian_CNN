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

        # # Initialize weights similar to the purely excitatory model
        # # TODO separate weight distribution initialisation for inhibitory and excitatory
        # weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        # initial_weights = weight_range * torch.abs(
        #     torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        # # Make inhibitory weights negative
        # initial_weights[self.excitatory_channels:] *= -1
        # self.weight = nn.Parameter(initial_weights)

        # Define weight range based on the same distribution
        weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        # Initialize excitatory weights
        excitatory_weights = weight_range * torch.abs(
            torch.randn((self.excitatory_channels, in_channels // groups, *self.kernel_size))
        )
        # Initialize inhibitory weights (same distribution, but will be made negative)
        inhibitory_weights = weight_range * torch.abs(
            torch.randn((out_channels - self.excitatory_channels, in_channels // groups, *self.kernel_size))
        )
        inhibitory_weights *= -1  # Make inhibitory weights negative
        # Concatenate excitatory and inhibitory weights
        initial_weights = torch.cat((excitatory_weights, inhibitory_weights), dim=0)
        # Assign to the module's weight parameter
        self.weight = nn.Parameter(initial_weights)


        # weight_range = 25 / math.sqrt(in_channels * kernel_size * kernel_size)
        # initial_weights = weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size))
        # initial_weights[:self.excitatory_channels] = torch.abs(initial_weights[:self.excitatory_channels])
        # initial_weights[self.excitatory_channels:] = -torch.abs(initial_weights[self.excitatory_channels:])
        # self.weight = nn.Parameter(initial_weights)

        # Initialize signs for excitatory and inhibitory neurons
        self.sign = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.sign[self.excitatory_channels:] = -1

        self.register_buffer('delta_w', torch.zeros_like(self.weight))

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

    def apply_weights(self, x, w):
        """
		This function provides the logic for combining input x and weight w
		"""
        # w = self.apply_lebesgue_norm(self.weight)
        x = symmetric_pad(x, self.padding)
        return F.conv2d(x, w, None, self.stride, 0, self.dilation, groups=self.groups)

    def compute_activation(self, x):
        w = self.weight * self.sign
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        if self.presynaptic_weights: w = self.compute_presynaptic_competition(w)
        y = self.act(self.apply_weights(x, w))
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
        if self.mode not in [self.MODE_BASIC_HEBBIAN, self.MODE_SOFTWTA,
                             self.MODE_HARDWT]:
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

            # Separate WTA for excitatory and inhibitory channels
            y_exc = y[:, :self.excitatory_channels]
            y_inh = y[:, self.excitatory_channels:]

            # WTA for excitatory channels
            y_exc_flat = y_exc.transpose(0, 1).reshape(self.excitatory_channels, -1)
            win_neurons_exc = torch.argmax(y_exc_flat, dim=0)
            wta_mask_exc = F.one_hot(win_neurons_exc, num_classes=self.excitatory_channels).float()
            wta_mask_exc = wta_mask_exc.transpose(0, 1).view(self.excitatory_channels, batch_size, height_out,
                                                             width_out).transpose(0, 1)
            y_wta_exc = y_exc * wta_mask_exc

            # WTA for inhibitory channels
            y_inh_flat = y_inh.transpose(0, 1).reshape(self.inhibitory_channels, -1)
            win_neurons_inh = torch.argmax(torch.abs(y_inh_flat), dim=0)
            wta_mask_inh = F.one_hot(win_neurons_inh, num_classes=self.inhibitory_channels).float()
            wta_mask_inh = wta_mask_inh.transpose(0, 1).view(self.inhibitory_channels, batch_size, height_out,
                                                             width_out).transpose(0, 1)
            y_wta_inh = y_inh * wta_mask_inh

            # Combine excitatory and inhibitory WTA results
            y_wta = torch.cat([y_wta_exc, y_wta_inh], dim=1)
            # Compute yx using conv2d
            yx = F.conv2d(x.transpose(0, 1), y_wta.transpose(0, 1), padding=0,
                          stride=self.dilation, dilation=self.stride).transpose(0, 1)
            # Compute y * w
            yu = torch.sum(y_wta, dim=(0, 2, 3)).view(-1, 1, 1, 1)
            yw = yu * weight
            # Compute update
            update = yx - yw

            # Normalization
            update.div_(torch.abs(update).amax() + 1e-8)
            self.delta_w += update

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


    @torch.no_grad()
    def local_update(self):
        # Apply update while respecting weight signs
        new_weight = self.weight + self.lr * self.alpha * self.delta_w * self.sign
        # Ensure weights maintain their sign
        new_weight *= self.sign
        # Update weights
        self.weight.copy_(new_weight)
        # Reset delta_w
        self.delta_w.zero_()
