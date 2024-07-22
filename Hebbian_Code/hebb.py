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
    MODE_HWTA = 'hwta'
    MODE_LATERAL_INHIBITION = "lateral"
    MODE_BCM = 'bcm'

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 w_nrm=True, bias=False, act=nn.Identity(),
                 mode=MODE_BASIC_HEBBIAN, k=1, patchwise=True,
                 contrast=1., uniformity=False, alpha=0., wta_competition='similarity_filter',
                 lateral_competition="filter",
                 lateral_inhibition_strength=0.01, top_k=1, prune_rate = 0):
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
        # self.act = self.cos_sim2d
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
        self.lateral_learning_rate = lateral_inhibition_strength  # Adjust as needed
        self.lebesgue_p = 2

        self.prune_rate = prune_rate  # 99% of connections are pruned

        self.lr_scheduler_config = {'lr': 0.1, 'adaptive': True}
        self.lr_adaptive = self.lr_scheduler_config['adaptive']
        if self.lr_adaptive:
            print("Adaptive learning rate")
            self.register_buffer('lr', torch.ones_like(self.weight))
        else:
            print("Stable learning rate")
            self.lr = self.lr_scheduler_config['lr']
            print(self.lr)

        self.reset()

    def reset(self):
        if self.lr_adaptive:
            self.update_lr()

    def generate_mask(self):
        return torch.bernoulli(torch.full_like(self.weight, 1 - self.prune_rate))

    def update_lr(self):
        if self.lr_adaptive:
            norm = self.get_radius()
            lr_amplitude = self.lr_scheduler_config['lr']
            power_lr = self.lr_scheduler_config.get('power_lr', 1)
            lr = lr_amplitude * torch.pow(torch.abs(norm - torch.ones_like(norm)) + 1e-10, power_lr)
            self.lr = lr.view(-1, 1, 1, 1).expand_as(self.weight)

    def get_radius(self):
        weight = self.weight.view(self.weight.shape[0], -1)
        return torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)

    def cos_sim2d(self, x):
        # print(x.shape)
        # print(w.shape)
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

    def apply_lebesgue_norm(self, w):
        return torch.sign(w) * torch.abs(w) ** (self.lebesgue_p - 1)

    def apply_weights(self, x, w):
        """
		This function provides the logic for combining input x and weight w
		"""
        w = self.apply_lebesgue_norm(self.weight)
        return torch.conv2d(x, w, bias=self.bias, stride=self.stride)

    def compute_activation(self, x):
        w = self.weight
        if self.w_nrm: w = normalize(w, dim=(1, 2, 3))
        y = self.act(self.apply_weights(x, w))
        # For cosine similarity activation
        # y = self.act(x)
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
                             self.MODE_WTA, self.MODE_LATERAL_INHIBITION, self.MODE_BCM, self.MODE_HWTA]:
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
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]
                batch_size, hw, in_features = x_unf.shape
                _, h, w, out_channels = y_unf.shape
                # Reshape y_unf
                y_unf = y_unf.reshape(batch_size * hw, out_channels)
                # Compute normalization factor
                y_sum = y_unf.sum(dim=0, keepdim=True)
                y_sum[y_sum == 0] = 1.0  # Avoid division by zero
                # Normalize y
                y_normalized = y_unf / y_sum
                # Compute the update using Grossberg's Instar Rule
                weight = self.weight.view(out_channels, -1)  # Shape: [out_channels, C*k*k]
                # Compute x_avg (average input for each output channel)
                x_avg = torch.mm(y_normalized.t(), x_unf.reshape(-1, in_features))
                # Compute the update
                update = x_avg - weight
                # Reshape the update to match the weight shape
                self.delta_w += update.reshape_as(self.weight)

        if self.mode == self.MODE_WTA:
            with torch.no_grad():
                # Unfold the input tensor
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]
                batch_size, hw, in_features = x_unf.shape
                _, h, w, out_channels = y_unf.shape
                y_unf_flat = y_unf.reshape(batch_size * hw, out_channels)

                if self.wta_competition == 'filter':
                    topk_values, _ = y_unf_flat.topk(self.top_k, dim=-1, largest=True, sorted=False)
                    y_mask = (y_unf_flat >= topk_values[..., -1, None]).float()
                elif self.wta_competition == 'spatial':
                    topk_values, _ = y_unf_flat.topk(self.top_k, dim=0, largest=True, sorted=False)
                    y_mask = (y_unf_flat >= topk_values[-1, :]).float()
                elif self.wta_competition == 'combined':
                    topk_filter, _ = y_unf_flat.topk(self.top_k, dim=-1, largest=True, sorted=False)
                    topk_spatial, _ = y_unf_flat.topk(self.top_k, dim=0, largest=True, sorted=False)
                    y_mask = ((y_unf_flat >= topk_filter[..., -1, None]) | (y_unf_flat >= topk_spatial[-1, :])).float()
                elif self.wta_competition in ['similarity_filter', 'similarity_spatial']:
                    similarity_cos = self.cos_sim2d(x)
                    similarity = similarity_cos.permute(0, 2, 3, 1).reshape(-1, self.out_channels)
                    # similarity = torch.matmul(x_unf.reshape(-1, in_features), weight_unf.t())
                    if self.wta_competition == 'similarity_filter':
                        topk_similarity, _ = similarity.topk(self.top_k, dim=-1, largest=True, sorted=False)
                        y_mask = (similarity >= topk_similarity[..., -1, None]).float()
                    else:  # similarity_spatial
                        topk_similarity, _ = similarity.topk(self.top_k, dim=0, largest=True, sorted=False)
                        y_mask = (similarity >= topk_similarity[-1, :]).float()
                else:
                    raise ValueError(f"Unknown WTA competition mode: {self.wta_competition}")

                self.binary_mask = self.generate_mask()
                # Apply mask and normalize
                # y_masked = y_unf_flat * y_mask
                # Global layer Normalization
                y_sum = y_mask.sum(dim=0, keepdim=True)
                y_sum[y_sum == 0] = 1.0  # Avoid division by zero
                # y_normalized = y_masked / y_sum
                # Local Normalization
                # self.y_avg = self.y_avg * 0.9 + y_masked.mean(dim=0) * 0.1  # Exponential moving average
                # y_local_norm = y_masked / (self.y_avg + 1e-5)  # Add small epsilon to prevent division by zero
                # Initialize lateral weights if they don't exist
                # y_inhibited = self.combined_lateral_inhibition(y_normalized, out_channels, hw)
                # Compute x_avg using inhibited activations
                x_avg = torch.mm(y_mask.t(), x_unf.reshape(-1, in_features))/ y_sum.t()
                # Mask for winning neurons only
                winner_mask = (y_sum > 0).float().view(self.out_channels, 1)
                # Compute the update: Apply the winner_mask to ensure non-winners do not update their weights
                update = winner_mask * (x_avg - self.weight.view(self.out_channels, -1))
                # L1 normalization of weights
                nc = torch.abs(update).amax()
                update.div_(nc + 1e-30)
                # Reshape the update to match the weight shape
                self.delta_w += (update.reshape_as(self.weight)*self.binary_mask)

        if self.mode == self.MODE_BCM:
            with torch.no_grad():
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]

                batch_size, hw, in_features = x_unf.shape
                _, h, w, out_channels = y_unf.shape
                y_unf_flat = y_unf.reshape(batch_size * hw, out_channels)
                # Update average activity for BCM
                y_avg_new = y_unf_flat.mean(dim=0)  # Shape: [out_channels]
                self.y_avg = self.y_avg + self.alpha_bcm * (y_avg_new - self.y_avg)
                # Compute BCM learning rule
                theta = self.y_avg ** 2  # Shape: [out_channels]
                y_bcm = y_unf_flat * (y_unf_flat - theta)  # Shape: [batch_size*H*W, out_channels]
                # Normalize y_bcm
                y_sum = y_bcm.sum(dim=0, keepdim=True)
                y_sum[y_sum == 0] = 1.0  # Avoid division by zero
                y_normalized = y_bcm / y_sum
                y_inhibited = self.combined_lateral_inhibition(y_normalized, out_channels, hw)
                # Compute x_avg (average input for each output channel)
                x_avg = torch.mm(y_inhibited.t(), x_unf.reshape(-1, in_features))
                # Compute the update
                update = x_avg - self.weight.view(out_channels, -1)
                nc = torch.abs(update).amax()
                update.div_(nc + 1e-30)
                # Reshape the update to match the weight shape
                self.delta_w += update.reshape_as(self.weight)

        if self.mode == self.MODE_HWTA:
            with torch.no_grad():
                x_unf = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
                x_unf = x_unf.permute(0, 2, 1)  # Shape: [batch_size, H*W, C*k*k]
                y_unf = y.permute(0, 2, 3, 1).contiguous()  # Shape: [batch_size, H, W, out_channels]
                batch_size, hw, in_features = x_unf.shape
                _, h, _, out_channels = y_unf.shape
                y_unf_flat = y_unf.reshape(batch_size * hw, out_channels)
                # Soft WTA using softmax
                temperature = 1.0  # You can adjust this value to control the "softness"
                y_swta = F.softmax(y_unf_flat / temperature, dim=1)
                # Compute yx (correlation between input and SWTA output)
                yx = torch.mm(y_swta.t(), x_unf.reshape(-1, in_features))
                # Compute yu (sum of element-wise product of SWTA and pre-activation)
                pre_x = self.compute_activation(x).permute(0, 2, 3, 1).reshape(batch_size * hw, out_channels)
                yu = torch.sum(y_swta * pre_x, dim=0)
                # Compute update
                w = self.weight.view(out_channels, -1)
                update = yx - yu.view(-1, 1) * w
                # Normalization
                nc = torch.abs(update).amax()
                update.div_(nc + 1e-30)
                self.delta_w += update.reshape_as(self.weight)

    def lateral_inhibition_within_filter(self, y_normalized, out_channels, hw):
        # y_normalized shape is [50176, 96], so hw = 50176 and out_channels = 96
        y_reshaped = y_normalized.t()  # Shape: [96, 50176]
        # Compute coactivation within each filter
        coactivation = torch.mm(y_reshaped, y_reshaped.t())  # Shape: [96, 96]
        # Update lateral weights
        if not hasattr(self, 'lateral_weights_filter'):
            self.lateral_weights_filter = torch.zeros_like(coactivation)
        lateral_update = self.lateral_learning_rate * (coactivation - self.lateral_weights_filter)
        self.lateral_weights_filter += lateral_update
        # Apply inhibition
        inhibition = torch.mm(self.lateral_weights_filter, y_reshaped)
        y_inhibited = F.relu(y_reshaped - inhibition)
        return y_inhibited.t()  # Shape: [50176, 96]

    def lateral_inhibition_same_patch(self, y_normalized, out_channels, hw):
        # y_normalized shape is already [hw, out_channels] = [50176, 96]
        y_reshaped = y_normalized
        # Compute coactivation for neurons looking at the same patch
        coactivation = torch.mm(y_reshaped.t(), y_reshaped)  # Shape: [96, 96]
        # Update lateral weights
        if not hasattr(self, 'lateral_weights_patch'):
            self.lateral_weights_patch = torch.zeros_like(coactivation)
        lateral_update = self.lateral_learning_rate * (coactivation - self.lateral_weights_patch)
        self.lateral_weights_patch += lateral_update
        # Apply inhibition
        inhibition = torch.mm(y_reshaped, self.lateral_weights_patch)
        y_inhibited = F.relu(y_reshaped - inhibition)
        return y_inhibited  # Shape: [50176, 96]

    def combined_lateral_inhibition(self, y_normalized, out_channels, hw):
        # Apply inhibition within filter
        y_inhibited_filter = self.lateral_inhibition_within_filter(y_normalized, out_channels, hw)
        # Apply inhibition for same patch
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
            self.weight.grad = self.lr*(-self.alpha * self.delta_w)
        else:
            self.weight.grad = self.lr*((1 - self.alpha) * self.weight.grad - self.alpha * self.delta_w)
        self.delta_w.zero_()
        if self.lr_adaptive:
            self.update_lr()
