import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

import params as P


class ZCAWhitening:
    def __init__(self, epsilon=1e-1):
        self.epsilon = epsilon
        self.zca_matrix = None
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor, transpose=True, dataset: str = "CIFAR10"):
        path = os.path.join("zca_data", dataset, f"{dataset}_zca.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            saved_data = torch.load(path, map_location='cpu')
            self.zca_matrix = saved_data['zca']
            self.mean = saved_data['mean']
            self.std = saved_data['std']
        except:
            if transpose and x.dim() == 4:
                x = x.permute(0, 3, 1, 2)

            x = x.reshape(x.shape[0], -1)
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True)
            x = (x - self.mean) / (self.std + self.epsilon)

            cov = torch.mm(x.T, x) / (x.shape[0] - 1)
            u, s, v = torch.svd(cov)

            inv_sqrt_s = torch.diag(1.0 / torch.sqrt(s + self.epsilon))
            self.zca_matrix = torch.mm(torch.mm(u, inv_sqrt_s), u.T)

            torch.save({'zca': self.zca_matrix, 'mean': self.mean, 'std': self.std}, path)

    def transform(self, x: torch.Tensor):
        if self.zca_matrix is None:
            raise ValueError("ZCA matrix not computed. Call fit() first.")

        original_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = (x - self.mean) / (self.std + self.epsilon)
        x_whitened = torch.mm(x, self.zca_matrix)
        return x_whitened.reshape(original_shape)


def whitening_zca(x: torch.Tensor, transpose=True, dataset: str = "CIFAR10"):
    zca = ZCAWhitening()
    zca.fit(x, transpose, dataset)
    return zca.transform(x)


class ZCATransformation:
    def __init__(self, zca):
        self.zca = zca

    def __call__(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x_whitened = self.zca.transform(x)
        return x_whitened.squeeze(0) if x_whitened.shape[0] == 1 else x_whitened


def visualize_zca_effect(original_data, whitened_data, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for i in range(num_samples):
        # Normalize data to [0, 1] range
        orig = original_data[i].permute(1, 2, 0)
        orig = (orig - orig.min()) / (orig.max() - orig.min())

        whit = whitened_data[i].permute(1, 2, 0)
        whit = (whit - whit.min()) / (whit.max() - whit.min())

        axes[0, i].imshow(orig)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        axes[1, i].imshow(whit)
        axes[1, i].axis('off')
        axes[1, i].set_title('Whitened')

    plt.tight_layout()
    plt.show()


def check_covariance(data):
    data_flat = data.reshape(data.shape[0], -1)
    cov_matrix = torch.cov(data_flat.T)
    diag_mean = cov_matrix.diag().mean().item()
    off_diag_mean = (cov_matrix - torch.diag(cov_matrix.diag())).abs().mean().item()
    print(f"Mean of diagonal elements: {diag_mean:.6f}")
    print(f"Mean of off-diagonal elements: {off_diag_mean:.6f}")
    print(f"Ratio of off-diagonal to diagonal: {off_diag_mean / diag_mean:.6f}")

def check_normalization(data):
    data_flat = data.reshape(data.shape[0], -1)
    mean = data_flat.mean().item()
    std = data_flat.std().item()
    print(f"Mean: {mean:.6f}")
    print(f"Standard deviation: {std:.6f}")
    print(f"Min: {data_flat.min().item():.6f}")
    print(f"Max: {data_flat.max().item():.6f}")


def get_data(dataset='cifar10', root='datasets', batch_size=32, num_workers=0, whiten_lvl=None):
    trn_set, tst_set = None, None
    if dataset == 'cifar10':
        trn_set = CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=T.ToTensor())
        tst_set = CIFAR10(root=os.path.join(root, dataset), train=False, download=True, transform=T.ToTensor())
    else:
        raise NotImplementedError("Dataset {} not supported.".format(dataset))

    zca = None
    if whiten_lvl is not None:
        print("Data whitening")
        zca = ZCAWhitening(epsilon=whiten_lvl)
        all_data = torch.cat([torch.tensor(trn_set.data), torch.tensor(tst_set.data)], dim=0)
        all_data = all_data.float() / 255.0  # Normalize data to [0, 1]
        zca.fit(all_data, transpose=True, dataset=dataset)

        # Create a temporary loader with only ToTensor transform for visualization
        temp_transform = T.Compose([T.Resize(32), T.ToTensor()])
        temp_dataset = CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=temp_transform)
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # Get a batch for visualization
        original_batch, _ = next(iter(temp_loader))
        whitened_batch = zca.transform(original_batch)

        # print("\nVisualization of ZCA effect:")
        # visualize_zca_effect(original_batch, whitened_batch)

        print("\nCovariance check:")
        print("Before whitening:")
        check_covariance(original_batch)
        print("\nAfter whitening:")
        check_covariance(whitened_batch)

        print("\nNormalization check:")
        print("Before whitening:")
        check_normalization(original_batch)
        print("\nAfter whitening:")
        check_normalization(whitened_batch)

        # Now apply the full transform including ZCA to the datasets
        full_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize(32),
            T.ToTensor(),
            ZCATransformation(zca),
        ])
        trn_set.transform = full_transform
        tst_set.transform = full_transform
    else:
        print("No ZCA")
        temp_transform = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        trn_set.transform = temp_transform
        tst_set.transform = temp_transform
    trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=False, num_workers=P.NUM_WORKERS)
    tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, num_workers=P.NUM_WORKERS)

    return trn_loader, tst_loader, zca


# Example usage:
if __name__ == "__main__":
    trn_loader, tst_loader, zca = get_data(dataset='cifar10', batch_size=64, whiten_lvl=1e-1)
    print("Data loaders and ZCA whitening prepared.")