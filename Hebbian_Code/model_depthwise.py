import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

# from hebb import HebbianConv2d
# from hebb_ex_in import HebbianConv2d
# from hebb_ffi import HebbianConv2d
# from hebb_depthwise import HebbianDepthConv2d

from hebb_abs import HebbianConv2d
from hebb_abs_depthwise import HebbianDepthConv2d



import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import islice
import umap
import wandb
import seaborn as sns
import networkx as nx

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
    def __init__(self, hebb_params=None, version="soft"):
        super(Net_Depthwise, self).__init__()

        if hebb_params is None: hebb_params = default_hebb_params
        if version == "softhebb":
            # A single Depthwise convolutional layer
            self.bn1 = nn.BatchNorm2d(3, affine=False)
            self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **hebb_params,
                                       padding=2, t_invert=1)
            self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
            self.activ1 = Triangle(power=0.7)

            self.bn2 = nn.BatchNorm2d(96, affine=False)
            self.conv2 = HebbianDepthConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, **hebb_params,
                                            t_invert=0.65, padding=1)
            self.bn_point2 = nn.BatchNorm2d(96, affine=False)
            self.conv_point2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=1, stride=1, **hebb_params,
                                             t_invert=0.65, padding=0)
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

        elif version == "hardhebb":
            # A single Depthwise convolutional layer
            self.bn1 = nn.BatchNorm2d(3, affine=False)
            self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **hebb_params,
                                       padding=0, t_invert=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.activ1 = Triangle(power=0.7)

            self.bn2 = nn.BatchNorm2d(96, affine=False)
            self.conv2 = HebbianDepthConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, **hebb_params,
                                            t_invert=0.65, padding=0)
            self.bn_point2 = nn.BatchNorm2d(96, affine=False)
            self.conv_point2 = HebbianConv2d(in_channels=96, out_channels=384, kernel_size=1, stride=1,
                                             **hebb_params,t_invert=0.65, padding=0)
            # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.activ2 = Triangle(power=1.4)

            self.bn3 = nn.BatchNorm2d(384, affine=False)
            self.conv3 = HebbianDepthConv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1,
                                            **hebb_params,
                                            t_invert=0.25, padding=0)
            self.bn_point3 = nn.BatchNorm2d(384, affine=False)
            self.conv_point3 = HebbianConv2d(in_channels=384, out_channels=1536, kernel_size=1, stride=1,
                                             **hebb_params,t_invert=0.25, padding=0)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.activ3 = Triangle(power=1.)

            self.flatten = nn.Flatten()
            # Final fully-connected layer classifier
            self.fc1 = nn.Linear(38400, 10) # 38400 30700
            self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 38400)
            self.dropout = nn.Dropout(0.5)

        elif version == "lagani":
            # A single Depthwise convolutional layer
            self.bn1 = nn.BatchNorm2d(3, affine=False)
            self.conv1 = HebbianConv2d(in_channels=3, out_channels=96, kernel_size=5, stride=1, **hebb_params,
                                       padding=0, t_invert=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.activ1 = Triangle(power=1.)

            self.bn2 = nn.BatchNorm2d(96, affine=False)
            self.conv2 = HebbianDepthConv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, **hebb_params,
                                            t_invert=0.65, padding=0)
            self.bn_point2 = nn.BatchNorm2d(96, affine=False)
            self.conv_point2 = HebbianConv2d(in_channels=96, out_channels=128, kernel_size=1, stride=1, **hebb_params,
                                             t_invert=0.65, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.activ2 = Triangle(power=1.)

            self.bn3 = nn.BatchNorm2d(128, affine=False)
            self.conv3 = HebbianDepthConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, **hebb_params,
                                            t_invert=0.25, padding=0)
            self.bn_point3 = nn.BatchNorm2d(128, affine=False)
            self.conv_point3 = HebbianConv2d(in_channels=128, out_channels=192, kernel_size=1, stride=1, **hebb_params,
                                             t_invert=0.25, padding=0)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.activ3 = Triangle(power=1.)

            self.bn4 = nn.BatchNorm2d(192, affine=False)
            self.conv4 = HebbianDepthConv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, **hebb_params,
                                            t_invert=0.25, padding=0)
            self.bn_point4 = nn.BatchNorm2d(192, affine=False)
            self.conv_point4 = HebbianConv2d(in_channels=192, out_channels=256, kernel_size=1, stride=1, **hebb_params,
                                             t_invert=0.25, padding=0)
            # self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.activ4 = Triangle(power=1.)

            self.flatten = nn.Flatten()
            # Final fully-connected layer classifier
            self.fc1 = nn.Linear(1728, 10)
            self.fc1.weight.data = 0.11048543456039805 * torch.rand(10, 1728)
            self.dropout = nn.Dropout(0.5)

    def forward_features(self, x):
        x = self.pool1(self.activ1(self.conv1(self.bn1(x))))
        return x

    def features_extract(self, x):
        # SoftHebb
        x = self.forward_features(x)
        x = self.activ2(self.conv_point2(self.bn_point2(self.conv2(self.bn2(x)))))
        x = self.pool3(self.activ3(self.conv_point3(self.bn_point3(self.conv3(self.bn3(x))))))
        # Lagani
        # x = self.forward_features(x)
        # x = self.activ2(self.conv_point2(self.bn_point2(self.conv2(self.bn2(x)))))
        # x = self.pool3(self.activ3(self.conv_point3(self.bn_point3(self.conv3(self.bn3(x))))))
        # x = self.activ4(self.conv4(self.bn4(x)))
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

                # # New 1x1 Convolution Structure Visualization
                # ax = axes[idx, 2]
                # num_out, num_in = weight_matrix.shape
                #
                # # Create a bipartite graph layout
                # pos = {}
                # for i in range(num_in):
                #     pos[f'in_{i}'] = (0, i)
                # for i in range(num_out):
                #     pos[f'out_{i}'] = (1, i)
                #
                # # Draw nodes
                # for node, (x, y) in pos.items():
                #     color = 'lightblue' if 'in_' in node else 'lightgreen'
                #     ax.scatter(x, y, c=color, s=100)
                #     ax.annotate(node, (x, y), xytext=(5 if 'in_' in node else -5, 0),
                #                 textcoords='offset points', ha='left' if 'in_' in node else 'right', va='center')
                #
                # # Draw edges
                # max_weight = np.max(np.abs(weight_matrix))
                # for i in range(num_out):
                #     for j in range(num_in):
                #         weight = weight_matrix[i, j]
                #         if abs(weight) > 0.1 * max_weight:  # Only draw significant connections
                #             ax.plot([0, 1], [j, i], color='red' if weight > 0 else 'blue',
                #                     alpha=min(1, abs(weight) / max_weight), linewidth=1.5)
                #
                # ax.set_title(f'{name} 1x1 Conv Structure')
                # ax.set_xlim(-0.1, 1.1)
                # ax.set_ylim(min(num_in, num_out) - 0.5, -0.5)
                # ax.axis('off')
                #
                # # Add text explanation
                # text = (
                #     f"This plot shows the structure of a 1x1 conv with {num_in} input and {num_out} output channels.\n"
                #     "Blue nodes: Input channels\n"
                #     "Green nodes: Output channels\n"
                #     "Red lines: Positive weights\n"
                #     "Blue lines: Negative weights\n"
                #     "Line opacity indicates weight strength (darker = stronger).")
                # ax.text(0.5, -0.1, text, transform=ax.transAxes, ha='center', va='center', fontsize=8, wrap=True)

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
        layer = getattr(self, layer_name)
        # Directly access the weight attributes
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
                        if i >= 20:  # Inhibitory
                            # Invert and scale the inhibitory weights to [0, 1] range for visualization
                            # filter_img = -filter_img
                            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
                            # Apply a blue tint to distinguish inhibitory filters
                            # filter_img = filter_img * np.array([0.5, 0.5, 1.0])
                        else:  # Excitatory
                            # Clip excitatory weights to [0, 1] range
                            filter_img = np.clip(filter_img, 0, 1)
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
            receptive_field = (receptive_field - receptive_field.min()) / (
                        receptive_field.max() - receptive_field.min() + 1e-8)
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
