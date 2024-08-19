import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from hebb import HebbianConv2d
from hebb_depthwise import HebbianDepthConv2d

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import islice
import umap
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

class Net_Depthwise(nn.Module):
    def __init__(self, hebb_params=None):
        super(Net_Depthwise, self).__init__()

        if hebb_params is None: hebb_params = default_hebb_params

        # A single Depthwise convolutional layer
        self.bn1 = nn.BatchNorm2d(3, affine=False)
        self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **hebb_params, padding=2, t_invert=1)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ1 = Triangle(power=0.7)


        self.bn2 = nn.BatchNorm2d(96, affine=False)
        self.conv2 = HebbianDepthConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, **hebb_params,
                                   t_invert=0.65, padding=1)
        self.bn_point2 = nn.BatchNorm2d(96, affine=False)
        self.conv_point2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=1, stride=1, **hebb_params, t_invert=0.65, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.activ2 = Triangle(power=1.4)


        self.bn3 = nn.BatchNorm2d(384, affine=False)
        self.conv3 = HebbianDepthConv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, **hebb_params,
                                   t_invert=0.25, padding=1)
        self.bn_point3 = nn.BatchNorm2d(384, affine=False)
        self.conv_point3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=1, stride=1, **hebb_params,
                                         t_invert=0.25, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.activ3 = Triangle(power=1.)

        self.flatten = nn.Flatten()
        # Final fully-connected layer classifier
        self.fc1 = nn.Linear(24576, 10)
        self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 24576)
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        x = self.forward_features(x)
        x = self.pool2(self.activ2(self.conv_point2(self.bn_point2(self.conv2(self.bn2(x))))))
        x = self.pool3(self.activ3(self.conv_point3(self.bn_point3(self.conv3(self.bn3(x))))))
        return x

    def forward(self, x):
        x = self.features_extract(x)
        x = self.flatten(x)
        x = self.fc1(self.dropout(x))
        return x

    def plot_grid(self, tensor, path, num_rows=5, num_cols=5, layer_name=""):
        # Ensure we're working with the first 25 filters (or less if there are fewer)
        tensor = tensor[:25]
        # Normalize the tensor
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
        weights = getattr(self, layer_name).weight.data
        self.plot_grid(weights, save_path, layer_name=layer_name)

    def visualize_receptive_fields(self, layer_name, dataloader, num_neurons=10, num_batches=10, save_path=None):
        self.eval()
        device = next(self.parameters()).device
        layer = getattr(self, layer_name)
        # Create a figure to display the receptive fields
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f'Receptive Fields of {layer_name}')
        # Hook to capture the output of the specified layer
        def hook_fn(module, input, output):
            self.activations = output
        handle = layer.register_forward_hook(hook_fn)
        # Compute gradients for the first `num_neurons` neurons
        receptive_fields = torch.zeros(num_neurons, *next(iter(dataloader))[0].shape[1:], device=device)
        for inputs, _ in tqdm(islice(dataloader, num_batches), desc="Computing receptive fields", total=num_batches):
            inputs = inputs.to(device)
            inputs.requires_grad_()
            # Perform a forward pass to get the activations
            self.activations = None  # Reset activations
            self(inputs)
            activations = self.activations
            for i in range(num_neurons):
                self.zero_grad()
                activation = activations[:, i].sum()
                activation.backward(retain_graph=True)
                # Accumulate gradients to average receptive field
                receptive_fields[i] += inputs.grad.abs().mean(dim=0)
                inputs.grad.zero_()
        receptive_fields /= num_batches
        for i in range(num_neurons):
            # Normalize the receptive field for visualization
            receptive_field = receptive_fields[i].cpu().numpy()
            receptive_field = (receptive_field - receptive_field.min()) / (receptive_field.max() - receptive_field.min() + 1e-8)
            # Plot the receptive field
            ax = axes[i // 5, i % 5]
            if receptive_field.shape[0] == 3:
                # For RGB images, use all channels
                ax.imshow(np.transpose(receptive_field, (1, 2, 0)))
            else:
                # For grayscale or other number of channels, use the first channel
                ax.imshow(receptive_field[0], cmap='viridis')
            ax.set_title(f'Neuron {i}')
            ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close(fig)
        # Remove the hook
        handle.remove()

    def visualize_in_input_space(self, dataloader, num_batches=10, n_neighbors=15, min_dist=0.1, n_components=2):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get the weight vectors
        weights = self.conv1.weight.detach().cpu().numpy().reshape(self.conv1.out_channels, -1)
        # Initialize lists to store flattened input data and labels
        input_data_flat = []
        labels_list = []
        # Iterate through the dataloader
        with torch.no_grad():
            for i, (data, labels) in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break
                data = data.to(device)
                # Extract features if possible, otherwise use raw data
                if hasattr(self, 'features_extract'):
                    features = self.extract_features(data)
                else:
                    features = data
                # Flatten input data and add to list
                batch_flat = features.view(features.size(0), -1).cpu().numpy()
                input_data_flat.append(batch_flat)
                labels_list.append(labels.cpu().numpy())
        # Concatenate all batches
        input_data_flat = np.vstack(input_data_flat)
        labels = np.concatenate(labels_list)
        # Pad weights to match input dimension
        input_dim = input_data_flat.shape[1]
        weight_dim = weights.shape[1]
        padded_weights = np.pad(weights, ((0, 0), (0, input_dim - weight_dim)), mode='constant')
        # Combine padded weights and input data
        combined_data = np.vstack([padded_weights, input_data_flat])
        # Normalize data before UMAP
        scaler = StandardScaler()
        combined_data_normalized = scaler.fit_transform(combined_data)
        # Apply UMAP
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
        embedded_data = reducer.fit_transform(combined_data_normalized)
        # Separate embedded weights and input data
        embedded_weights = embedded_data[:self.conv1.out_channels]
        embedded_inputs = embedded_data[self.conv1.out_channels:]
        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedded_inputs[:, 0], embedded_inputs[:, 1], c=labels, alpha=0.5, cmap='tab10')
        plt.colorbar(scatter, label='Class Labels')
        plt.scatter(embedded_weights[:, 0], embedded_weights[:, 1], c='red', marker='x', s=100,
                    label='Weight Vectors')
        plt.title('Input Data, Labels, and Weight Vectors in 2D UMAP Space')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.show()
