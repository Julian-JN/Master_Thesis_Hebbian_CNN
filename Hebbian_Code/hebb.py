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

    MODE_SWTA = 'swta'
    MODE_HPCA = 'hpca'
    MODE_CONTRASTIVE = 'contrastive'
    MODE_BASIC_HEBBIAN = 'basic_hebbian'
    MODE_WTA = 'wta'
    MODE_LATERAL_INHIBITION = "lateral"

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 w_nrm=True, bias=False, act=nn.Identity(),
                 mode=MODE_BASIC_HEBBIAN, k=1, patchwise=True,
                 contrast=1., uniformity=False, alpha=0., wta_competition='similarity_filter', lateral_competition="filter",
                 lateral_inhibition_strength = 0.1):
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
                             self.MODE_WTA, self.MODE_LATERAL_INHIBITION]:
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
                print("Shapes")
                if self.patchwise:
                    print(cr.reshape(r.shape[0], -1).shape)
                    print(x_unf.shape)
                    print((cr.reshape(r.shape[0], -1).matmul(x_unf)).shape)
                    print("Dec")
                    print(cr.reshape(r.shape[0], -1).sum(1, keepdim=True).shape)
                    print(self.weight.reshape(self.weight.shape[0],-1).shape)
                    dec = cr.reshape(r.shape[0], -1).sum(1, keepdim=True) * self.weight.reshape(self.weight.shape[0],-1)
                    print(dec.shape)
                    self.delta_w += (cr.reshape(r.shape[0], -1).matmul(x_unf) - dec).reshape_as(self.weight)
                    print(self.delta_w.shape)
                else:
                    krn = torch.eye(len(self.weight[0]), device=x.device, dtype=x.dtype).view(len(self.weight[0]),
                                                                                              self.weight.shape[1],
                                                                                              *self.kernel_size)
                    dec = torch.conv_transpose2d(cr.sum(dim=1, keepdim=True) * self.weight.reshape(1, 1, self.weight.shape[0],
                                                                -1).permute(2, 3,0, 1),krn, stride=self.stride)
                    self.delta_w += (cr.reshape(r.shape[0], -1).matmul(x_unf) - F.unfold(dec, kernel_size=self.kernel_size,
                                                                stride=self.stride).sum(dim=-1)).reshape_as(self.weight)

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
                            kernel_size=self.kernel_size, stride=self.stride).sum(dim=-1)).reshape_as(self.weight)

        if self.mode == self.MODE_BASIC_HEBBIAN:
            with torch.no_grad():
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
                y_unf = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
                self.delta_w += y_unf.t().matmul(x_unf).reshape_as(self.weight)

        if self.mode == self.MODE_WTA:
            print("WTA")
            with torch.no_grad():
                # Unfold the input tensor
                print(x.shape)
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                print(x_unf.shape)
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]
                print(y_unf.shape)

                if self.wta_competition == 'filter':
                    y_max, _ = y_unf.max(dim=-1, keepdim=True)
                    y_unf = (y_unf == y_max).float() * y_unf
                elif self.wta_competition == 'spatial':
                    y_max, _ = y_unf.reshape(-1, y_unf.size(-1)).max(dim=0, keepdim=True)
                    y_unf = (y_unf == y_max.view(1, 1, 1, -1)).float() * y_unf
                elif self.wta_competition == 'combined':
                    y_max_filter, _ = y_unf.max(dim=-1, keepdim=True)
                    y_max_spatial, _ = y_unf.reshape(-1, y_unf.size(-1)).max(dim=0, keepdim=True)
                    y_unf = ((y_unf == y_max_filter) | (y_unf == y_max_spatial.view(1, 1, 1, -1))).float() * y_unf
                elif self.wta_competition == 'similarity_filter' or self.wta_competition == 'similarity_spatial':
                    # Compute similarity between weights and inputs
                    weight_unf = self.weight.view(self.weight.size(0), -1)
                    similarity = torch.matmul(x_unf, weight_unf.t())  # Shape: [batch_size, H*W, out_channels]

                    if self.wta_competition == 'similarity_filter':
                        max_similarity, _ = similarity.max(dim=-1, keepdim=True)
                        similarity_mask = (similarity == max_similarity).float()
                    else:  # similarity_spatial
                        max_similarity, _ = similarity.reshape(-1, similarity.size(-1)).max(dim=0, keepdim=True)
                        similarity_mask = (similarity == max_similarity.view(1, 1, -1)).float()

                    y_unf = similarity_mask * y_unf.view(*similarity.shape)

                # Reshape for update computation
                x_unf_reshaped = x_unf.reshape(-1, x_unf.size(-1))  # Shape: [batch_size*H*W, C*k*k]
                y_unf_reshaped = y_unf.reshape(-1, y_unf.size(-1))  # Shape: [batch_size*H*W, out_channels]

                print(x_unf_reshaped.shape)
                print(y_unf_reshaped.shape)
                print(self.weight.shape)

                # Compute x - w
                weight_reshaped = self.weight.view(self.weight.size(0), -1)  # Shape: [out_channels, C*k*k]
                print(weight_reshaped.shape)
                x_minus_w = x_unf_reshaped.unsqueeze(1) - weight_reshaped.unsqueeze(0) # x_minus_w shape: [12544, 64, 75]
                print(x_minus_w.shape)

                # Compute weighted average
                y_weighted = y_unf_reshaped / (y_unf_reshaped.sum(dim=0, keepdim=True) + 1e-6)
                print(y_weighted.shape)

                # Compute the update using bmm
                update = torch.einsum('ijk,ij->jk', x_minus_w, y_weighted)
                print(update.shape)  # Should be [64, 75]
                update_reshape = update.reshape_as(self.weight)
                print(update_reshape.shape)
                # Add update to delta_w
                self.delta_w += update_reshape
                # Decay term calculation
                # print("Shapes")
                # print(x_unf.shape)
                # print(self.weight.shape)
                # if self.patchwise:
                #     dec = y_unf.t().sum(1, keepdim=True) * self.weight.view(self.weight.size(0), -1)
                #     self.delta_w += (y_unf.t().matmul(x_unf) - dec).view_as(self.weight)
                # else:
                #     krn = torch.eye(self.weight.size(1), device=x.device, dtype=x.dtype).view(self.weight.size(1), 1,
                #                                                                               *self.kernel_size)
                #     dec = torch.conv_transpose2d((y_unf.sum(1, keepdim=True).view_as(y)), krn, stride=self.stride)
                #     dec_unf = F.unfold(dec, kernel_size=self.kernel_size, stride=self.stride).sum(dim=-1)
                #     self.delta_w += (y_unf.t().matmul(x_unf) - dec_unf).view_as(self.weight)

        if self.mode == self.MODE_LATERAL_INHIBITION:
            with torch.no_grad():
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1).reshape(-1, x_unf.size(1))
                y_unf = y.permute(0, 2, 3, 1).reshape(-1, y.size(1))
                # Hebbian update
                hebbian_update = y_unf.t().matmul(x_unf).reshape_as(self.weight)
                # Anti-Hebbian lateral inhibition
                if self.lateral_inhibition_mode == 'filter':
                    # Inhibition between neurons of the same filter
                    inhibition = y_unf.t().matmul(y_unf)
                    inhibition = inhibition - torch.diag(torch.diag(inhibition))  # Remove self-inhibition
                    lateral_update = -self.lateral_inhibition_strength * inhibition.matmul(
                        self.weight.view(self.out_channels, -1)).view_as(self.weight)
                elif self.lateral_inhibition_mode == 'spatial':
                    # Inhibition between different neurons looking at the same patch
                    y_patch = y.permute(0, 2, 3, 1).reshape(-1, self.out_channels)
                    inhibition = y_patch.t().matmul(y_patch)
                    inhibition = inhibition - torch.diag(torch.diag(inhibition))  # Remove self-inhibition
                    lateral_update = -self.lateral_inhibition_strength * inhibition.unsqueeze(-1).unsqueeze(-1) * self.weight
                # Combine Hebbian and anti-Hebbian updates
                self.delta_w += hebbian_update + lateral_update

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
