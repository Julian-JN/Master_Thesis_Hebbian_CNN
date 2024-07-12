import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x, dim=None):
    nrm = (x ** 2).sum(dim=dim, keepdim=True) ** 0.5
    nrm[nrm == 0] = 1.
    return x / nrm


class HebbianConv2d(nn.Module):
    """
	A 2d convolutional layer that learns through Hebbian plasticity
	"""

    MODE_SWTA = 'swta'
    MODE_HPCA = 'hpca'
    MODE_CONTRASTIVE = 'contrastive'
    MODE_BASIC_HEBBIAN = 'basic'
    MODE_WTA = 'wta'
    MODE_LATERAL_INHIBITION = "lateral"
    MODE_BCM = 'bcm'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 w_nrm=True, bias=False, act=nn.Identity(),
                 mode=MODE_BASIC_HEBBIAN, k=1, patchwise=True,
                 contrast=1., uniformity=False, alpha=0., wta_competition='similarity_filter',
                 lateral_competition="filter",
                 lateral_inhibition_strength=0.1, top_k=3):
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

        self.k = k
        self.top_k = top_k
        self.register_buffer('y_avg', torch.zeros(out_channels))  # Average activity for BCM
        self.alpha_bcm = 0.5

        self.patchwise = patchwise
        self.contrast = contrast
        self.uniformity = uniformity
        self.alpha = alpha
        self.wta_competition = wta_competition
        self.lateral_inhibition_mode = lateral_competition
        self.lateral_inhibition_strength = lateral_inhibition_strength

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
        y = self.compute_activation(x)
        if self.training and self.alpha != 0: self.compute_update(x, y)
        return y

    def compute_update(self, x, y):
        """
		This function implements the logic that computes local plasticity rules from input x and output y. The
		resulting weight update is stored in buffer self.delta_w for later use.
		"""
        if self.mode not in [self.MODE_SWTA, self.MODE_HPCA, self.MODE_CONTRASTIVE, self.MODE_BASIC_HEBBIAN,
                             self.MODE_WTA, self.MODE_LATERAL_INHIBITION, self.MODE_BCM]:
            raise NotImplementedError(
                "Learning mode {} unavailable for {} layer".format(self.mode, self.__class__.__name__))

        if self.mode == self.MODE_SWTA:
            with torch.no_grad():
                # Logic for swta-type learning
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
                r = (y * self.k).softmax(dim=1).permute(1, 0, 2, 3)
                r_sum = r.abs().sum(dim=(1, 2, 3), keepdim=True)
                r_sum = r_sum + (r_sum == 0).float()  # Prevent divisions by zero
                c = r.abs() / r_sum
                cr = c * r
                if self.patchwise:
                    dec = cr.reshape(r.shape[0], -1).sum(1, keepdim=True) * self.weight.reshape(self.weight.shape[0],
                                                                                                -1)
                    self.delta_w += (cr.reshape(r.shape[0], -1).matmul(x_unf) - dec).reshape_as(self.weight)
                else:
                    krn = torch.eye(len(self.weight[0]), device=x.device, dtype=x.dtype).view(len(self.weight[0]),
                                                                                              self.weight.shape[1],
                                                                                              *self.kernel_size)
                    dec = torch.conv_transpose2d(
                        cr.sum(dim=1, keepdim=True) * self.weight.reshape(1, 1, self.weight.shape[0], -1).permute(2, 3,
                                                                                                                  0, 1),
                        krn, stride=self.stride)
                    self.delta_w += (
                            cr.reshape(r.shape[0], -1).matmul(x_unf) - F.unfold(dec, kernel_size=self.kernel_size,
                                                                                stride=self.stride).sum(
                        dim=-1)).reshape_as(self.weight)

        if self.mode == self.MODE_HPCA:
            with torch.no_grad():
                # Logic for hpca-type learning
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
                r = y.permute(1, 0, 2, 3)
                c = 1 / (r.shape[1] * r.shape[2] * r.shape[3])
                cr = c * r
                l = (torch.arange(self.weight.shape[0], device=x.device, dtype=x.dtype).unsqueeze(0).repeat(
                    self.weight.shape[0], 1) <= torch.arange(self.weight.shape[0], device=x.device,
                                                             dtype=x.dtype).unsqueeze(1)).to(dtype=x.dtype)
                if self.patchwise:
                    dec = (cr.reshape(r.shape[0], -1).matmul(r.reshape(r.shape[0], -1).transpose(-2, -1)) * l).matmul(
                        self.weight.reshape(self.weight.shape[0], -1))
                    self.delta_w += (cr.reshape(r.shape[0], -1).matmul(x_unf) - dec).reshape_as(self.weight)
                else:
                    r, cr = r.permute(2, 3, 0, 1), cr.permute(2, 3, 0, 1)
                    dec = torch.conv_transpose2d(
                        (cr.matmul(r.transpose(-2, -1)) * l.unsqueeze(0).unsqueeze(1)).permute(3, 2, 0, 1), self.weight,
                        stride=self.stride)
                    self.delta_w += (cr.permute(2, 3, 0, 1).reshape(r.shape[2], -1).matmul(x_unf) - F.unfold(dec,
                                                                                                             kernel_size=self.kernel_size,
                                                                                                             stride=self.stride).sum(
                        dim=-1)).reshape_as(self.weight)
                # print(self.delta_w)

        if self.mode == self.MODE_BASIC_HEBBIAN:
            with torch.no_grad():
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
                y_unf = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
                self.delta_w += y_unf.t().matmul(x_unf).reshape_as(self.weight)

        if self.mode == self.MODE_WTA:
            with torch.no_grad():
                # Unfold the input tensor
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]

                if self.wta_competition == 'filter':
                    topk_values, _ = y_unf.topk(self.top_k, dim=-1, largest=True, sorted=False)
                    y_unf = (y_unf >= topk_values[..., -1, None]).float() * y_unf
                elif self.wta_competition == 'spatial':
                    y_unf_flat = y_unf.reshape(-1, y_unf.size(-1))  # Shape: [batch_size * H * W, out_channels]
                    topk_values, _ = y_unf_flat.topk(self.top_k, dim=0, largest=True, sorted=False)
                    topk_values = topk_values[-1, :]  # Keep only the k-th largest value for each channel
                    y_unf = (y_unf >= topk_values).float() * y_unf
                elif self.wta_competition == 'combined':
                    topk_values_filter, _ = y_unf.topk(self.top_k, dim=-1, largest=True, sorted=False)
                    topk_values_filter = topk_values_filter[..., -1, None]
                    y_unf_flat = y_unf.reshape(-1, y_unf.size(-1))  # Shape: [batch_size * H * W, out_channels]
                    topk_values_spatial, _ = y_unf_flat.topk(self.top_k, dim=0, largest=True, sorted=False)
                    topk_values_spatial = topk_values_spatial[-1,
                                          :]  # Keep only the k-th largest value for each channel
                    y_unf = ((y_unf >= topk_values_filter) | (y_unf >= topk_values_spatial)).float() * y_unf

                elif self.wta_competition == 'similarity_filter' or self.wta_competition == 'similarity_spatial':
                    # Compute similarity between weights and inputs
                    weight_unf = self.weight.view(self.weight.size(0), -1)
                    similarity = torch.matmul(x_unf, weight_unf.t())
                    if self.wta_competition == 'similarity_filter':
                        topk_similarity, _ = similarity.topk(self.top_k, dim=-1, largest=True, sorted=False)
                        topk_similarity = topk_similarity[..., -1, None]
                    else:  # similarity_spatial
                        similarity_flat = similarity.reshape(-1, similarity.size(-1))
                        topk_similarity, _ = similarity_flat.topk(self.top_k, dim=0, largest=True, sorted=False)
                        topk_similarity = topk_similarity[-1, :]
                    similarity_mask = (similarity >= topk_similarity).float()
                    y_unf = similarity_mask * y_unf.view(*similarity.shape)
                    y_unf = similarity_mask * y_unf.view(*similarity.shape)

                # Reshape for update computation
                x_unf_reshaped = x_unf.reshape(-1, x_unf.size(-1))  # Shape: [batch_size*H*W, C*k*k]
                y_unf_reshaped = y_unf.reshape(-1, y_unf.size(-1))  # Shape: [batch_size*H*W, out_channels]
                # Compute x - w
                weight_reshaped = self.weight.view(self.weight.size(0), -1)  # Shape: [out_channels, C*k*k]
                x_minus_w = x_unf_reshaped.unsqueeze(1) - weight_reshaped.unsqueeze(
                    0)  # x_minus_w shape: [12544, 64, 75]
                # Compute weighted average
                y_weighted = y_unf_reshaped / (y_unf_reshaped.sum(dim=0, keepdim=True) + 1e-6)
                # Compute the update using bmm
                update = torch.einsum('ijk,ij->jk', x_minus_w, y_weighted)
                update_reshape = update.reshape_as(self.weight)
                # Add update to delta_w
                self.delta_w += update_reshape

        if self.mode == self.MODE_BCM:
            with torch.no_grad():  # Use no_grad to reduce memory usage
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]
                x_unf_reshaped = x_unf.reshape(-1, x_unf.size(-1))  # Shape: [batch_size*H*W, C*k*k]
                y_unf_reshaped = y_unf.reshape(-1, y_unf.size(-1))  # Shape: [batch_size*H*W, out_channels]
                weight_reshaped = self.weight.view(self.weight.size(0), -1)  # Shape: [out_channels, C*k*k]
                # Compute x - w
                x_minus_w = x_unf_reshaped.unsqueeze(1) - weight_reshaped.unsqueeze(
                    0)  # Shape: [batch_size*H*W, out_channels, C*k*k]
                # Update average activity for BCM
                y_avg_new = y_unf_reshaped.mean(dim=0)  # Shape: [out_channels]
                self.y_avg = self.y_avg + self.alpha_bcm * (y_avg_new - self.y_avg)
                # Compute BCM learning rule
                theta = (self.y_avg ** 2).unsqueeze(0)  # Shape: [1, out_channels]
                y_bcm = y_unf_reshaped * (y_unf_reshaped - theta)  # Shape: [batch_size*H*W, out_channels]
                # Compute the update using einsum
                update = torch.einsum('ij,ik->jk', x_unf_reshaped, y_bcm)  # Shape: [C*k*k, out_channels]
                update_reshape = update.t().reshape_as(self.weight)
                self.delta_w += update_reshape

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
