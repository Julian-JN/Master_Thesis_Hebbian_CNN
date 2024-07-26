import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from hebb import HebbianConv2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from itertools import islice
import umap



default_hebb_params = {'mode': HebbianConv2d.MODE_SOFTWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 0.}


class Triangle(nn.Module):
    def __init__(self, power: float = 0.7, inplace: bool = False, eps: float = 1e-6):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        centered = input - torch.mean(input, dim=1, keepdim=True)
        positive = F.relu(centered, inplace=self.inplace)
        negative = F.relu(-centered, inplace=self.inplace)
        result = torch.abs(positive - negative + self.eps) ** self.power
        return result * torch.sign(positive - negative)

class Net(nn.Module):
    def __init__(self, hebb_params=None):
        super().__init__()

        if hebb_params is None: hebb_params = default_hebb_params

        # A single convolutional layer
        self.conv1 = HebbianConv2d(3, 96, 5, 1, **hebb_params)
        self.bn1 = nn.BatchNorm2d(96, affine=False)
        # Aggregation stage
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pool = nn.MaxPool2d(2)

        # Final fully-connected 2-layer classifier
        hidden_shape = self.get_hidden_shape()
        self.conv2 = HebbianConv2d(96, 128, 3, 1, **hebb_params, prune_rate=0.99)
        self.bn2 = nn.BatchNorm2d(128, affine=False)
        self.fc1 = nn.Linear(128 * 12 * 12, 300)
        self.fc2 = nn.Linear(300, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization for convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def get_hidden_shape(self):
        self.eval()
        with torch.no_grad(): out = self.forward_features(torch.ones([1, 3, 32, 32], dtype=torch.float32)).shape[1:]
        return out

    def forward_features(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.bn1(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.bn2(torch.relu_(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(torch.dropout(x.reshape(x.shape[0], x.shape[1]), p=0.1, train=self.training))
        return x

    def forward_hebbian(self, x):
        x = self.forward_features(x)
        x = self.bn2(torch.relu(self.conv2(x)))
        return x


    def hebbian_train(self, dataloader, device):
        self.train()
        for inputs, _ in tqdm(dataloader, ncols=80):
            inputs = inputs.to(device)
            _ = self.forward_hebbian(inputs)  # Only forward pass through conv layers to trigger Hebbian updates
            for layer in [self.conv1, self.conv2]:
                if isinstance(layer, HebbianConv2d):
                    layer.local_update()

    def plot_grid(self, tensor, path, num_rows=3, num_cols=4, layer_name=""):
        # Ensure we're working with the first 12 filters (or less if there are fewer)
        tensor = tensor[:12]
        # Normalize the tensor
        # tensor = torch.sigmoid((tensor - tensor.mean()) / tensor.std())
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
        # Move to CPU and convert to numpy
        tensor = tensor.cpu().detach().numpy()
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        fig.suptitle(f'First 12 Filters of {layer_name}')
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

        plt.tight_layout()
        fig.savefig(path, bbox_inches='tight')
        plt.show()
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
                if hasattr(self, 'extract_features'):
                    features = self.forward_features(data)
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
