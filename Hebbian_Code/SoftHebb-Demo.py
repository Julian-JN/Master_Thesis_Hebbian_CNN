"""
Demo single-file script to train a ConvNet on CIFAR10 using SoftHebb, an unsupervised, efficient and bio-plausible
learning algorithm
"""
import math
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.optim.lr_scheduler import StepLR
import data

from torchvision.transforms import Normalize
from tqdm import tqdm

# import torchvision
# import eagerpy as ep
# import foolbox
# from foolbox import PyTorchModel
# from foolbox.attacks.base import T, get_criterion, raise_if_kwargs
# from foolbox.attacks.gradient_descent_base import BaseGradientDescent, normalize_lp_norms, uniform_l2_n_balls
# from foolbox.criteria import Misclassification, TargetedMisclassification
# from foolbox.devutils import flatten
# from foolbox.distances import l2
# from foolbox.models.base import Model
# from foolbox.types import Bounds
# from typing import Union, Any

torch.manual_seed(0)


class SoftHebbConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            t_invert: float = 12,
    ) -> None:
        """
        Simplified implementation of Conv2d learnt with SoftHebb; an unsupervised, efficient and bio-plausible
        learning algorithm.
        This simplified implementation omits certain configurable aspects, like using a bias, groups>1, etc. which can
        be found in the full implementation in hebbconv.py
        """
        super(SoftHebbConv2d, self).__init__()
        assert groups == 1, "Simple implementation does not support groups > 1."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = 'reflect'
        self.F_padding = (padding, padding, padding, padding)
        weight_range = 25 / math.sqrt((in_channels / groups) * kernel_size * kernel_size)
        self.weight = nn.Parameter(weight_range * torch.randn((out_channels, in_channels // groups, *self.kernel_size)))
        self.t_invert = torch.tensor(t_invert)

    def forward(self, x):
        x = F.pad(x, self.F_padding, self.padding_mode)  # pad input
        # perform conv, obtain weighted input u \in [B, OC, OH, OW]
        weighted_input = F.conv2d(x, self.weight, None, self.stride, 0, self.dilation, self.groups)

        if self.training:
            # ===== find post-synaptic activations y = sign(u)*softmax(u, dim=C), s(u)=1 - 2*I[u==max(u,dim=C)] =====
            # Post-synaptic activation, for plastic update, is weighted input passed through a softmax.
            # Non-winning neurons (those not with the highest activation) receive the negated post-synaptic activation.
            batch_size, out_channels, height_out, width_out = weighted_input.shape
            # Flatten non-competing dimensions (B, OC, OH, OW) -> (OC, B*OH*OW)
            flat_weighted_inputs = weighted_input.transpose(0, 1).reshape(out_channels, -1)
            # Compute the winner neuron for each batch element and pixel
            flat_softwta_activs = torch.softmax(self.t_invert * flat_weighted_inputs, dim=0)
            flat_softwta_activs = - flat_softwta_activs  # Turn all postsynaptic activations into anti-Hebbian
            win_neurons = torch.argmax(flat_weighted_inputs, dim=0)  # winning neuron for each pixel in each input
            competing_idx = torch.arange(flat_weighted_inputs.size(1))  # indeces of all pixel-input elements
            # Turn winner neurons' activations back to hebbian
            flat_softwta_activs[win_neurons, competing_idx] = - flat_softwta_activs[win_neurons, competing_idx]
            softwta_activs = flat_softwta_activs.view(out_channels, batch_size, height_out, width_out).transpose(0, 1)
            # ===== compute plastic update Δw = y*(x - u*w) = y*x - (y*u)*w =======================================
            # Use Convolutions to apply the plastic update. Sweep over inputs with postynaptic activations.
            # Each weighting of an input pixel & an activation pixel updates the kernel element that connected them in
            # the forward pass.
            yx = F.conv2d(
                x.transpose(0, 1),  # (B, IC, IH, IW) -> (IC, B, IH, IW)
                softwta_activs.transpose(0, 1),  # (B, OC, OH, OW) -> (OC, B, OH, OW)
                padding=0,
                stride=self.dilation,
                dilation=self.stride,
                groups=1
            ).transpose(0, 1)  # (IC, OC, KH, KW) -> (OC, IC, KH, KW)
            # sum over batch, output pixels: each kernel element will influence all batches and output pixels.
            yu = torch.sum(torch.mul(softwta_activs, weighted_input), dim=(0, 2, 3))
            delta_weight = yx - yu.view(-1, 1, 1, 1) * self.weight
            delta_weight.div_(torch.abs(delta_weight).amax() + 1e-30)  # Scale [min/max , 1]
            delta_weight = delta_weight.contiguous()  # Make contiguous
            # Store in grad to be used with common optimizers
            self.weight.grad = delta_weight
        return weighted_input


class DeepSoftHebb(nn.Module):
    def __init__(self):
        super(DeepSoftHebb, self).__init__()
        # block 1
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = SoftHebbConv2d(in_channels=3, out_channels=96, kernel_size=5, padding=2, t_invert=1,)
        self.activ1 = Triangle(power=0.7)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 2
        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = SoftHebbConv2d(in_channels=96, out_channels=384, kernel_size=3, padding=1, t_invert=0.65,)
        self.activ2 = Triangle(power=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # block 3
        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = SoftHebbConv2d(in_channels=384, out_channels=1536, kernel_size=3, padding=1, t_invert=0.25,)
        self.activ3 = Triangle(power=1.)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # block 4
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(24576, 10)
        self.classifier.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # block 1
        out = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        # block 2
        out = self.pool2(self.activ2(self.conv2(self.bn2(out))))
        # block 3
        out = self.pool3(self.activ3(self.conv3(self.bn3(out))))
        # block 4
        return self.classifier(self.dropout(self.flatten(out)))

    def features_extract(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        x = self.pool2(self.activ2(self.conv2(self.bn2(x))))
        x = self.pool3(self.activ3(self.conv3(self.bn3(x))))
        return x

    def plot_grid(self, tensor, path, num_rows=5, num_cols=5, layer_name=""):
        # Ensure we're working with the first 25 filters (or less if there are fewer)
        tensor = tensor[:25]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        # Move to CPU and convert to numpy
        tensor = tensor.cpu().detach().numpy()

        if tensor.shape[2] == 1 and tensor.shape[3] == 1:  # 1x1 convolution case
            out_channels, in_channels = tensor.shape[:2]
            fig = plt.figure(figsize=(14, 10))
            # Create a gridspec for the layout
            gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 3],
                                  left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            cbar_ax = fig.add_subplot(gs[:, 1])
            # Bar plot for average weights per filter
            avg_weights = tensor.mean(axis=(1, 2, 3))
            norm = plt.Normalize(vmin=avg_weights.min(), vmax=avg_weights.max())
            im1 = ax1.bar(range(out_channels), avg_weights, color=plt.cm.RdYlGn(norm(avg_weights)))
            ax1.set_xlabel('Filter Index')
            ax1.set_ylabel('Average Weight')
            ax1.set_title(f'Average Weights for 1x1 Kernels in {layer_name}')
            ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
            # Heatmap for detailed weight distribution
            im2 = ax2.imshow(tensor.reshape(out_channels, in_channels), cmap='RdYlGn', aspect='auto', norm=norm)
            ax2.set_xlabel('Input Channel')
            ax2.set_ylabel('Output Channel (Filter)')
            ax2.set_title('Detailed Weight Distribution')
            # Add colorbar to the right of both subplots
            fig.colorbar(im2, cax=cbar_ax, label='Normalized Weight Value')

        else:
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
            fig.suptitle(f'First 25 Filters of {layer_name}')
            for i, ax in enumerate(axes.flat):
                if i < tensor.shape[0]:
                    filter_img = tensor[i]
                    # Handle different filter shapes
                    if filter_img.shape[0] == 3:  # RGB filter (3, H, W)
                        filter_img = np.transpose(filter_img, (1, 2, 0))
                        # filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
                    elif filter_img.shape[0] == 1:  # Grayscale filter (1, H, W)
                        filter_img = filter_img.squeeze()
                    else:  # Multi-channel filter (C, H, W), take mean across channels
                        filter_img = np.mean(filter_img, axis=0)
                    ax.imshow(filter_img, cmap='viridis' if filter_img.ndim == 2 else None)
                    ax.set_title(f'Filter {i + 1}')
                ax.axis('off')
        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def visualize_filters(self, layer_name='conv1', save_path=None):
        weights = getattr(self, layer_name).weight.data
        self.plot_grid(weights, save_path, layer_name=layer_name)


class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power


class WeightNormDependentLR(optim.lr_scheduler._LRScheduler):
    """
    Custom Learning Rate Scheduler for unsupervised training of SoftHebb Convolutional blocks.
    Difference between current neuron norm and theoretical converged norm (=1) scales the initial lr.
    """

    def __init__(self, optimizer, power_lr, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.initial_lr_groups = [group['lr'] for group in self.optimizer.param_groups]  # store initial lrs
        self.power_lr = power_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        new_lr = []
        for i, group in enumerate(self.optimizer.param_groups):
            for param in group['params']:
                # difference between current neuron norm and theoretical converged norm (=1) scales the initial lr
                # initial_lr * |neuron_norm - 1| ** 0.5
                norm_diff = torch.abs(torch.linalg.norm(param.view(param.shape[0], -1), dim=1, ord=2) - 1) + 1e-10
                new_lr.append(self.initial_lr_groups[i] * (norm_diff ** self.power_lr)[:, None, None, None])
        return new_lr


class TensorLRSGD(optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, using a non-scalar (tensor) learning rate.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(-group['lr'] * d_p)
        return loss


class CustomStepLR(StepLR):
    """
    Custom Learning Rate schedule with step functions for supervised training of linear readout (classifier)
    """

    def __init__(self, optimizer, nb_epochs):
        threshold_ratios = [0.2, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.step_thresold = [int(nb_epochs * r) for r in threshold_ratios]
        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


def visualize_data_clusters(dataloader, model=None, method='tsne', dim=2, perplexity=30, n_neighbors=15, min_dist=0.1,
                            n_components=2, random_state=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_list = []
    labels_list = []
    if model is not None:
        model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            if model is not None:
                if hasattr(model, 'features_extract'):
                    features = model.features_extract(data)
                else:
                    features = model.conv1(data)
            else:
                features = data
            features = features.view(features.size(0), -1).cpu().numpy()
            features_list.append(features)
            labels_list.append(labels.numpy())
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=dim, perplexity=perplexity, n_iter=1000, random_state=random_state)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=dim, random_state=random_state)
    else:
        raise ValueError("Method must be either 'tsne' or 'umap'")
    projected_data = reducer.fit_transform(features_normalized)
    # Plotting
    if dim == 2:
        fig = plt.figure(figsize=(12, 10))
        scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, alpha=0.5, cmap='tab10')
        plt.colorbar(scatter, label='Class Labels')
        plt.title(f'CIFAR-10 Data Clusters using {method.upper()} (2D)')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
    elif dim == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2], c=labels, alpha=0.5,
                             cmap='tab10')
        fig.colorbar(scatter, label='Class Labels')
        ax.set_title(f'CIFAR-10 Data Clusters using {method.upper()} (3D)')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_zlabel(f'{method.upper()} Component 3')
    else:
        raise ValueError("dim must be either 2 or 3")
    plt.show()

# class ChooseNeuronFlat(torch.nn.Module):
#     def __init__(self, idx):
#         super(ChooseNeuronFlat, self).__init__()
#         self.idx = idx
#
#     def forward(self, x):
#         # return x[:, self.idx].view(-1)
#         selected_channel = x[:, self.idx, :, :]
#         # Flatten to a 1D tensor
#         return selected_channel.view(-1)


# class SingleMax:
#     def __init__(self, max_val: float, eps: float):
#         super().__init__()
#         self.max_val: float = max_val
#         self.eps: float = eps
#
#     def __call__(self, outputs):
#         if self.max_val is None:
#             is_adv = outputs > 0 | outputs <= 0
#         else:
#             is_adv = (self.max_val - outputs) < self.eps
#         return is_adv

class L2ProjGradientDescent:
    def __init__(self, steps, rel_stepsize, random_start=True):
        self.steps = steps
        self.rel_stepsize = rel_stepsize
        self.random_start = random_start

    def __call__(self, model, inputs, criterion, epsilon):
        if self.random_start:
            x = inputs + torch.randn_like(inputs) * 0.001 * epsilon
            x = torch.clamp(x, 0, 1)
        else:
            x = inputs.clone()
        x.requires_grad_(True)
        for _ in range(self.steps):
            model.zero_grad()
            outputs = model(x)
            loss = criterion(x, outputs).sum()  # Negative for ascent
            loss.backward()
            with torch.no_grad():
                gradients = x.grad
                print(gradients)
                stepsize = self.rel_stepsize * epsilon
                x += stepsize * gradients.sign()
                x = torch.clamp(x, 0, 1)  # Project to valid image range
            x.grad.zero_()
        return x.detach()

def attack_model(model, data, epsilons, iterations, step_size, criterion, device, random_start=True, nb_start=1):
    attack = L2ProjGradientDescent(steps=iterations, rel_stepsize=step_size, random_start=random_start)
    epsilon = epsilons[0] if isinstance(epsilons, (list, np.ndarray)) else epsilons
    best_input = None
    best_activation = float('-inf')
    for _ in range(nb_start):
        optimized_input = attack(model, data, criterion, epsilon)
        activation = model(optimized_input).max().item()
        if activation > best_activation:
            best_input = optimized_input
            best_activation = activation
    return best_input

def get_partial_model(model, target_layer):
    layers = []
    for layer in model.children():
        layers.append(layer)
        if layer == target_layer:
            break
    return torch.nn.Sequential(*layers)

# def visualize_filters(model, layer, num_filters=25, input_shape=(1, 3, 32, 32), step_size=0.1, iterations=500, random_start=True, nb_start=1):
#     model.eval()
#     partial_model = get_partial_model(model, layer)
#     partial_model = remove_padding(partial_model, layer)
#     receptive_field_size = calculate_receptive_field(partial_model, layer)
#     print(f"Receptive field size: {receptive_field_size}x{receptive_field_size}")
#     input_shape = (1, 3, receptive_field_size, receptive_field_size)
#
#     if hasattr(layer, 'out_channels'):
#         out_channels = layer.out_channels
#     else:
#         raise ValueError("The target layer does not have an 'out_channels' attribute.")
#
#     num_filters = min(num_filters, out_channels)
#     filter_images = []
#
#     for filter_idx in range(num_filters):
#         choose_neuron = ChooseNeuronFlat(filter_idx)
#         criterion = SingleMax(max_val=1e10, eps=1e-05)
#
#         data = torch.zeros(input_shape, device='cuda')
#         epsilons = np.array([255]) / 255  # Matching the original code's epsilon definition
#
#         optimized_input = attack_model(
#             model=torch.nn.Sequential(partial_model, choose_neuron).eval(),
#             data=data,
#             epsilons=epsilons,
#             iterations=iterations,
#             step_size=step_size,
#             criterion=criterion,
#             device='cuda',
#             random_start=random_start,
#             nb_start=nb_start)
#
#         optimized_image = optimized_input.cpu().squeeze(0).permute(1, 2, 0)
#         optimized_image = (optimized_image - optimized_image.min()) / (optimized_image.max() - optimized_image.min())
#         filter_images.append(optimized_image)
#
#     # Plot the filter visualizations in a grid
#     grid_size = int(num_filters ** 0.5) + (1 if num_filters ** 0.5 % 1 > 0 else 0)
#     fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
#     axes = axes.flatten()
#     for i in range(num_filters):
#         axes[i].imshow(filter_images[i].numpy())
#         axes[i].set_title(f'Filter {i + 1}')
#         axes[i].axis('off')
#     # Turn off unused subplots
#     for j in range(num_filters, len(axes)):
#         axes[j].axis('off')
#     plt.tight_layout()
#     plt.show()

def l2_normalize(x, epsilon=1.0):
    """Project x onto the L2 sphere with radius epsilon."""
    norm = torch.norm(x)
    return x * (epsilon / norm)

def calculate_receptive_field(model, target_layer):
    """
    Calculate the receptive field for a specific target Hebbian layer in the model.
    This will account for custom Hebbian layers similar to Conv2d.
    """
    current_rf = 1  # Start with a receptive field of 1x1
    current_stride = 1  # Start with stride 1
    current_kernel = 1  # Identity kernel to begin
    # Iterate through layers until we reach the target layer
    for layer in model.children():
        # Assuming your custom Hebbian layers have kernel_size and stride attributes
        if isinstance(layer, SoftHebbConv2d):  # Process only the SoftHebbConv2d layers
            kernel_size = layer.kernel_size[0]  # Assuming square kernels, take the first value of the pair
            stride = layer.stride[0]  # Assuming square strides, take the first value of the pair
            current_rf += (kernel_size - 1) * current_stride
            current_stride *= stride
        elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):  # Process pooling layers
            kernel_size = layer.kernel_size
            stride = layer.stride
            # Update receptive field for the pooling layer
            current_rf = current_rf + (kernel_size - 1) * current_stride
            current_stride *= stride  # Multiply by the pooling layer's stride
        if layer == target_layer:  # Stop when we reach the target layer
            break
    return current_rf


def get_layer_output(model, x, target_layer):
    """
    Forward pass through the model, stopping at the target custom Hebbian layer.
    """
    for layer in model.children():
        x = layer(x)  # Pass through each layer
        if layer == target_layer:
            return x  # Return the output of the target Hebbian layer
    raise ValueError(f"Target layer {target_layer} not found in the model.")


def remove_padding(model, target_layer):
    """Remove padding from layers before the target layer."""
    for layer in model.children():
        if isinstance(layer, (nn.Conv2d, SoftHebbConv2d)):
            layer.padding = (0, 0)
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d)):
            layer.padding = (0, 0)
        if layer == target_layer:
            break
    return model


class SingleMax:
    def __init__(self, max_val: float, eps: float):
        self.max_val = max_val
        self.eps = eps

    def __call__(self, outputs):
        if self.max_val is None:
            return outputs > 0
        else:
            return (self.max_val - outputs) < self.eps

def normalize_gradient(grad):
    """Normalize gradient to have L2 norm of 1."""
    norm = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1)
    return grad / (norm.view(-1, 1, 1, 1) + 1e-10)  # Add small epsilon to avoid division by zero

def get_random_start(x0, epsilon):
    batch_size, c, h, w = x0.shape
    r = torch.randn(batch_size, c * h * w, device=x0.device)
    r = r / r.norm(dim=1, keepdim=True)
    r = r.view_as(x0)
    return x0 + 0.00001 * torch.tensor(epsilon, device=x0.device) * r


def normalize_lp_norms(x, p=2):
    norms = x.view(x.shape[0], -1).norm(p=p, dim=1)
    return x / norms.view(-1, 1, 1, 1)


def project(x, x0, epsilon):
    delta = x - x0
    sphere = epsilon * normalize_lp_norms(delta, p=2).abs()
    sphere = sphere - sphere.min()
    if not sphere.max() > 0 and not sphere.min() < 0:
        return x0 + sphere
    else:
        return x0 + sphere / sphere.max()


def optimize_filter(model, layer, filter_idx, input_shape, step_size, iterations, random_start, criterion, epsilon):
    model.eval()
    x0 = torch.zeros(input_shape, device='cuda')
    if random_start:
        x = get_random_start(x0, epsilon)
    else:
        x = x0.clone()
    x.requires_grad_(True)
    for step in range(iterations):
        activation = get_layer_output(model, x, layer)
        loss = -activation[0, filter_idx].sum()  # maximally activates the filter as a whole
        if criterion(loss.item()):
            break  # Stop if the criterion is met
        loss.backward()
        with torch.no_grad():
            gradients = normalize_gradient(x.grad)  # Normalize gradients
            x.data = x.data + step_size * gradients
            x.data = project(x.data, x0, epsilon)
            x.data.clamp_(0, 1)  # Ensure values are in [0, 1] range
        x.grad.zero_()
    return x.detach(), -loss.item()


def visualize_filters(model, layer, num_filters=25, input_shape=(1, 3, 32, 32), step_size=0.001, iterations=500,
                      random_start=True, nb_start=1, l2_norm=True, max_val=1e10, eps=1e-5, epsilons=None):
    if epsilons is None:
        epsilons = torch.tensor([255.0]) / 255.0  # Default value as PyTorch tensor
    else:
        epsilons = torch.tensor(epsilons)  # Convert numpy array to PyTorch tensor
    model.eval()
    model = remove_padding(model, layer)
    receptive_field_size = calculate_receptive_field(model, layer)
    print(f"Receptive field size: {receptive_field_size}x{receptive_field_size}")
    input_shape = (1, 3, receptive_field_size, receptive_field_size)
    if hasattr(layer, 'out_channels'):
        out_channels = layer.out_channels
    else:
        raise ValueError("The target layer does not have an 'out_channels' attribute.")
    num_filters = min(num_filters, out_channels)
    filter_images = []
    criterion = SingleMax(max_val, eps)
    for filter_idx in range(num_filters):
        best_image = None
        best_activation = float('-inf')
        for start in range(nb_start):
            optimized_image, activation = optimize_filter(model, layer, filter_idx, input_shape, step_size, iterations,
                                                          random_start, criterion, epsilons)
            if activation > best_activation:
                print(f"New best Receptive Field found in Reboot {start}")
                best_activation = activation
                best_image = optimized_image
        optimized_image = best_image.cpu().squeeze(0).permute(1, 2, 0)
        optimized_image = (optimized_image - optimized_image.min()) / (optimized_image.max() - optimized_image.min())
        filter_images.append(optimized_image)
    # Plot the filter visualizations in a grid
    grid_size = int(num_filters ** 0.5) + (1 if num_filters ** 0.5 % 1 > 0 else 0)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    axes = axes.flatten()
    for i in range(num_filters):
        axes[i].imshow(filter_images[i].numpy())
        axes[i].set_title(f'Filter {i + 1}')
        axes[i].axis('off')
    # Turn off unused subplots
    for j in range(num_filters, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.show()

# Main training loop CIFAR10
if __name__ == "__main__":
    device = torch.device('cuda:0')
    model = DeepSoftHebb()
    model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameter Count Total: {num_parameters}")

    unsup_optimizer = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": -0.08, },  # SGD does descent, so set lr to negative
        {"params": model.conv2.parameters(), "lr": -0.005, },
        {"params": model.conv3.parameters(), "lr": -0.01, }
    ], lr=0)

    # unsup_optimizer = TensorLRSGD([
    #     {"params": model.conv1.parameters(), "lr": -0.1, },  # SGD does descent, so set lr to negative
    #     {"params": model.conv2.parameters(), "lr": -0.1, },
    #     {"params": model.conv3.parameters(), "lr": -0.1, }
    # ], lr=0)
    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)

    sup_optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    sup_lr_scheduler = CustomStepLR(sup_optimizer, nb_epochs=50)
    criterion = nn.CrossEntropyLoss()

    trn_set, tst_set, zca = data.get_data(dataset='cifar10', root='datasets', batch_size=64,
                                          whiten_lvl=None)

    # print("Visualizing Initial Filters")
    # model.visualize_filters('conv1', f'results/{"softhebb"}/conv1_filters_epoch_{1}.png')
    # model.visualize_filters('conv2', f'results/{"softhebb"}/conv2_filters_epoch_{1}.png')

    # Unsupervised training with SoftHebb
    running_loss = 0.0
    for i, data in enumerate(trn_set, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        # zero the parameter gradients
        unsup_optimizer.zero_grad()
        # forward + update computation
        with torch.no_grad():
            outputs = model(inputs)
        # optimize
        unsup_optimizer.step()
        unsup_lr_scheduler.step()

    print("Visualizing Filters")
    model.visualize_filters('conv1', f'results/{"softhebb"}/conv1_filters_epoch_{1}.png')
    model.visualize_filters('conv2', f'results/{"softhebb"}/conv2_filters_epoch_{1}.png')

    print("Visualizing Receptive fields")
    visualize_filters(model, model.conv1, num_filters=25)
    visualize_filters(model, model.conv2, num_filters=25)
    visualize_filters(model, model.conv3, num_filters=25)


    # set requires grad false and eval mode for all modules but classifier
    print("Classifier")
    unsup_optimizer.zero_grad()
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv3.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.conv3.eval()
    model.bn1.eval()
    model.bn2.eval()
    model.bn3.eval()

    print("Visualizing Class separation")
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2)
    for epoch in range(50):
        model.classifier.train()
        model.dropout.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trn_set, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            sup_optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            sup_optimizer.step()
            # compute training statistics
            running_loss += loss.item()
            if epoch % 10 == 0 or epoch == 49:
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        sup_lr_scheduler.step()
        # Evaluation on test set
        if epoch % 10 == 0 or epoch == 1:
            print(f'Accuracy of the network on the train images: {100 * correct // total} %')
            print(f'[{epoch + 1}] loss: {running_loss / total:.3f}')

            # on the test set
            model.eval()
            running_loss = 0.
            correct = 0
            total = 0
            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for data in tst_set:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = model(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()

            print(f'Accuracy of the network on the test images: {100 * correct / total} %')
            print(f'test loss: {running_loss / total:.3f}')