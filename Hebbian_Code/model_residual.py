import torch
import torch.nn as nn
import torch.nn.functional as F

from hebb import HebbianConv2d
from hebb_depthwise import HebbianDepthConv2d

import matplotlib.pyplot as plt
import numpy as np
import wandb
import seaborn as sns

torch.manual_seed(0)
default_hebb_params = {'mode': HebbianConv2d.MODE_SOFTWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 1.}


class Triangle(nn.Module):
    def __init__(self, power: float = 1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power

class LongSkipConnection(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_change=1, hebb_params=None):
        super(LongSkipConnection, self).__init__()
        if hebb_params is None:
            hebb_params = default_hebb_params

        self.conv = HebbianConv2d(in_channels, out_channels, kernel_size=1, stride=1, **hebb_params)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.spatial_change = spatial_change

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.spatial_change != 1:
            x = F.adaptive_avg_pool2d(x, output_size=x.size()[2:] // self.spatial_change)
        return x


class HebbianResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, hebb_params=None, t_invert=1., expansion_factor=6, act=1.):
        super(HebbianResidualBlock, self).__init__()

        if hebb_params is None:
            hebb_params = default_hebb_params

        # Calculate padding to maintain spatial dimensions
        # Doubt regarding when additional padding is required
        padding = (kernel_size - 1) // 2
        hidden_dim = in_channels * expansion_factor

        self.bn1 = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = HebbianConv2d(in_channels, hidden_dim, kernel_size=1, stride=1, **hebb_params, t_invert=t_invert,
                                        padding=0)

        self.bn2 = nn.BatchNorm2d(hidden_dim, affine=False)
        self.conv2 = HebbianDepthConv2d(hidden_dim, hidden_dim, kernel_size, stride=1, **hebb_params,
                                        t_invert=t_invert, padding=padding)

        self.bn3 = nn.BatchNorm2d(hidden_dim, affine=False)
        self.conv3 = HebbianConv2d(hidden_dim, out_channels, kernel_size=1, stride=1, **hebb_params, t_invert=t_invert,
                                   padding=0)

        self.activ = Triangle(power=act)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels, affine=False),
            HebbianConv2d(in_channels, out_channels, kernel_size=1, stride=1, **hebb_params, padding=0)
            )

    def forward(self, x):
        residual = x
        out = self.activ(self.conv1(self.bn1(x)))
        out = self.activ(self.conv2(self.bn2(out)))
        out = self.conv3(self.bn3(out))
        out += self.shortcut(residual)
        return self.activ(out)

class Net_Depthwise_Residual(nn.Module):
    def __init__(self, hebb_params=None):
        super(Net_Depthwise_Residual, self).__init__()

        if hebb_params is None:
            hebb_params = default_hebb_params

        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **hebb_params, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = Triangle(power=0.7)

        self.res1 = HebbianResidualBlock(96, 384, kernel_size=3, expansion_factor=4, hebb_params=hebb_params, t_invert=0.65, act=1.4)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)

        self.res2 = HebbianResidualBlock(384, 1536, kernel_size=3, expansion_factor=4, hebb_params=hebb_params, t_invert=0.25)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(24576, 10)  # Adjust this based on your input size
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.res1(x))
        x = self.pool3(self.res2(x))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        return x

    def plot_grid_ex_in(self, wee, wei, wie, path, num_rows=5, num_cols=5, layer_name=""):
        def normalize(tensor):
            max_abs = torch.max(torch.abs(tensor))
            return tensor / (max_abs + 1e-8)

        norm_wee = normalize(wee[:10])  # Take only first 10 filters
        norm_wei = normalize(wei[:10])  # Take only first 10 filters
        norm_wie = normalize(wie[-5:])  # Use all filters for WIE
        weight_types = [('WEE', norm_wee, 10), ('WEI', norm_wei, 10), ('WIE', norm_wie, norm_wie.shape[0])]

        if all(w.shape[2] == 1 and w.shape[3] == 1 for _, w, _ in weight_types):  # All 1x1 convolution case
            fig, axes = plt.subplots(3, 3, figsize=(20, 20))
            fig.suptitle(f'Filters of {layer_name} (1x1 convolutions)', fontsize=16)

            for idx, (name, weights, _) in enumerate(weight_types):
                weight_matrix = weights.squeeze().cpu().detach().numpy()

                # Heatmap (unchanged)
                ax = axes[idx, 0]
                im = ax.imshow(weight_matrix, cmap='coolwarm', aspect='auto')
                ax.set_title(f'{name} Weights Heatmap')
                ax.set_xlabel('Input Channel' if name != 'WIE' else 'Inhibitory Channel')
                ax.set_ylabel('Output Channel' if name != 'WIE' else 'Excitatory Channel')
                plt.colorbar(im, ax=ax)

                # Distribution plot (unchanged)
                ax = axes[idx, 1]
                sns.histplot(weight_matrix.flatten(), kde=True, ax=ax)
                ax.set_title(f'{name} Weights Distribution')
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Frequency')

        else:
            total_rows = sum(num for _, _, num in weight_types)
            fig, axes = plt.subplots(total_rows, 3, figsize=(15, 5 * total_rows))
            fig.suptitle(f'Filters of {layer_name}', fontsize=16)

            row = 0
            for name, weights, num_filters in weight_types:
                for i in range(num_filters):
                    if i < weights.shape[0]:
                        filter_img = weights[i].cpu().detach().numpy()
                        if name == 'WIE' and filter_img.shape[1] == 1 and filter_img.shape[2] == 1:
                            # Special case for 1x1 convolution in WIE
                            filter_img = filter_img.squeeze()

                            ax = axes[row, 0]
                            im = ax.imshow(filter_img.reshape(1, -1), cmap='YlOrRd', aspect='auto')
                            ax.set_title(f'{name} Filter {i + 1} Weights')
                            ax.set_xlabel('Inhibitory Channel')
                            ax.set_ylabel('Excitatory Channel')
                            plt.colorbar(im, ax=ax)

                            ax = axes[row, 1]
                            ax.hist(filter_img.ravel(), bins=20, color='skyblue', edgecolor='black')
                            ax.set_title(f'{name} Filter {i + 1} Distribution')
                            ax.set_xlabel('Weight Value')
                            ax.set_ylabel('Frequency')

                            ax = axes[row, 2]
                            ax.bar(range(len(filter_img)), filter_img, color='orange')
                            ax.set_title(f'{name} Filter {i + 1} Weights')
                            ax.set_xlabel('Inhibitory Channel')
                            ax.set_ylabel('Weight Value')

                        else:
                            if filter_img.shape[0] == 3:  # RGB filter
                                filter_img = np.transpose(filter_img, (1, 2, 0))
                            elif filter_img.shape[0] == 1:  # Grayscale filter
                                filter_img = filter_img.squeeze()
                            else:  # Multi-channel filter
                                filter_img = np.mean(filter_img, axis=0)

                            ax = axes[row, 0]
                            ax.imshow(filter_img, cmap='viridis' if filter_img.ndim == 2 else None)
                            ax.set_title(f'{name} Filter {i + 1}')
                            ax.axis('off')

                            ax = axes[row, 1]
                            ax.hist(filter_img.ravel(), bins=50)
                            ax.set_title(f'{name} Filter {i + 1} Distribution')

                            ax = axes[row, 2]
                            ax.imshow(np.abs(filter_img), cmap='hot')
                            ax.set_title(f'{name} Filter {i + 1} Magnitude')
                            ax.axis('off')
                    else:
                        # If there aren't enough filters, create empty plots
                        for j in range(3):
                            axes[row, j].axis('off')
                    row += 1

        plt.tight_layout()
        if path:
            fig.savefig(path, bbox_inches='tight')
        wandb.log({f'{layer_name} filters': wandb.Image(fig)})
        plt.close(fig)

    def visualize_filters_ex_in(self, layer_name='conv1', save_path=None):
        # Split the layer name by '.' to handle nested attributes
        attrs = layer_name.split('.')
        layer = self
        for attr in attrs:
            layer = getattr(layer, attr)
        wee = layer.weight_ee.data
        wei = layer.weight_ei.data
        wie = layer.weight_ie.data
        self.plot_grid_ex_in(wee, wei, wie, save_path, layer_name=layer_name)

    def plot_grid(self, tensor, path, num_rows=5, num_cols=5, layer_name=""):
        # Ensure we're working with the first 25 filters (or less if there are fewer)
        # tensor = tensor[:25]
        excitatory = tensor[:20]
        inhibitory = tensor[-5:]
        # Symmetric normalization for excitatory weights
        max_abs_exc = torch.max(torch.abs(excitatory))
        norm_exc = excitatory / (max_abs_exc + 1e-8)
        # Symmetric normalization for inhibitory weights
        max_abs_inh = torch.max(torch.abs(inhibitory))
        norm_inh = inhibitory / (max_abs_inh + 1e-8)
        tensor = torch.cat((norm_exc, norm_inh))
        # tensor = torch.cat((tensor[:20], tensor[-5:]))
        # Normalize the tensor
        # tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
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
                        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
                    elif filter_img.shape[0] == 1:  # Grayscale filter (1, H, W)
                        filter_img = filter_img.squeeze()
                    else:  # Multi-channel filter (C, H, W), take mean across channels
                        filter_img = np.mean(filter_img, axis=0)
                    ax.imshow(filter_img, cmap='viridis' if filter_img.ndim == 2 else None)
                    ax.set_title(f'Filter {i + 1}')
                ax.axis('off')

        if path:
            fig.savefig(path, bbox_inches='tight')
        wandb.log({f'{layer_name} filters': wandb.Image(fig)})
        plt.close(fig)

    def visualize_filters(self, layer_name='conv1', save_path=None):
        # Split the layer name by '.' to handle nested attributes
        attrs = layer_name.split('.')
        layer = self
        for attr in attrs:
            layer = getattr(layer, attr)
        weights = layer.weight.data
        self.plot_grid(weights, save_path, layer_name=layer_name)



