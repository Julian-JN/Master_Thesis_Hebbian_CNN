import torch
import torch.nn as nn

from hebb import HebbianConv2d
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm


default_hebb_params = {'mode': HebbianConv2d.MODE_SWTA, 'w_nrm': True, 'k': 50, 'act': nn.Identity(), 'alpha': 0.}

class Net(nn.Module):
	def __init__(self, hebb_params=None):
		super().__init__()

		if hebb_params is None: hebb_params = default_hebb_params

		# A single convolutional layer
		self.conv1 = HebbianConv2d(3, 96, 5, 1, **hebb_params)
		self.bn1 = nn.BatchNorm2d(96, affine=False)

		# Aggregation stage
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.pool = nn.MaxPool2d(2)

		# Final fully-connected 2-layer classifier
		hidden_shape = self.get_hidden_shape()
		self.conv2 = HebbianConv2d(96, 128, 3, 1, **hebb_params)
		self.bn2 = nn.BatchNorm2d(128, affine=False)
		self.fc3 = nn.Linear(128 * 12 * 12, 10)

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
		x = self.bn1(torch.relu(self.conv1(x)))
		x = self.pool(x)
		return x
	
	def forward(self, x):
		x = self.forward_features(x)
		x = self.bn2(torch.relu(self.conv2(x)))
		# print(x.shape)
		x = x.view(x.size(0), -1)  # This reshapes the tensor to (batch_size, 128 * k * k)
		x = self.fc3(x)
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

	def visualize_class_separation(self, dataloader, device, save_path=None):
		self.eval()
		features = []
		labels = []

		with torch.no_grad():
			for inputs, targets in dataloader:
				inputs = inputs.to(device)
				x = self.forward_features(inputs)
				x = self.bn2(torch.relu(self.conv2(x)))
				x = x.view(x.size(0), -1)
				features.append(x.cpu().numpy())
				labels.append(targets.numpy())

		features = np.concatenate(features, axis=0)
		labels = np.concatenate(labels, axis=0)

		tsne = TSNE(n_components=2, random_state=42)
		features_2d = tsne.fit_transform(features)

		plt.figure(figsize=(10, 8))
		scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10')
		plt.colorbar(scatter)
		plt.title('t-SNE visualization of features before the linear layer')
		if save_path:
			plt.savefig(save_path)
		plt.show()
	