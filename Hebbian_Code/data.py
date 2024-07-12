import os

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import params as P

# Transformations to be applied to the training data
T_trn = T.Compose([
    T.Resize(32),  # Resize shortest size of the image to a fixed size.
    # T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=30/360)], p=0.3), # Random color jittering
    # T.RandomApply([T.ColorJitter(saturation=1.)], p=0.1), # Randomly transform image saturation from nothing to full grayscale
    # T.RandomHorizontalFlip(), # Randomly flip image horizontally.
    # T.RandomVerticalFlip(), # Randomly flip image vertically.
    # T.RandomApply([T.Pad(16, padding_mode='reflect'), T.RandomRotation(10), T.CenterCrop(32)], p=0.3), # Random rotation
    # T.RandomApply([T.RandomCrop(32, padding=8, pad_if_needed=True, padding_mode='reflect')], p=0.3), # Random translation and final cropping with fixed size
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Transformations to be applied to the test data
T_tst = T.Compose([
    T.Resize(32),  # Resize shortest size of the image to a fixed size.
    # T.CenterCrop(32), # Center crop of fixed size
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_data(dataset='cifar10', root='datasets', batch_size=32, num_workers=0, whiten_lvl=None):
    trn_set, tst_set = None, None
    if dataset == 'cifar10':
        trn_set = DataLoader(CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=T_trn),
                             batch_size=batch_size, shuffle=True, num_workers=P.NUM_WORKERS)
        tst_set = DataLoader(CIFAR10(root=os.path.join(root, dataset), train=False, download=True, transform=T_tst),
                             batch_size=batch_size, shuffle=False, num_workers=P.NUM_WORKERS)
    else:
        raise NotImplementedError("Dataset {} not supported.".format(dataset))

    zca = None
    if whiten_lvl is not None:
        raise NotImplemented("Whitening not implemented.")

    return trn_set, tst_set, zca


def whiten(inputs, zca):
    raise NotImplemented("Whitening not implemented.")

# import os
#
# from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
# import torchvision.transforms as T
#
# import params as P
# import numpy as np
# import torch
#
#
# class ZCAWhitening:
#     def __init__(self, epsilon=1e-5):
#         self.epsilon = epsilon
#         self.mean = None
#         self.zca_matrix = None
#
#     def fit(self, X):
#         X = X.reshape(X.shape[0], -1)
#         self.mean = np.mean(X, axis=0)
#         X -= self.mean
#         cov_matrix = np.cov(X, rowvar=False)
#         U, S, _ = np.linalg.svd(cov_matrix)
#         self.zca_matrix = U @ np.diag(1.0 / np.sqrt(S + self.epsilon)) @ U.T
#
#     def whiten(self, X):
#         X = X.reshape(X.shape[0], -1)
#         X -= self.mean
#         return X @ self.zca_matrix
#
# class ZCATransformation:
#     def __init__(self, zca):
#         self.zca = zca
#
#     def __call__(self, x):
#         x_np = x.numpy().reshape(1, -1)
#         x_whitened = self.zca.whiten(x_np).reshape(3, 32, 32)
#         return torch.tensor(x_whitened, dtype=torch.float32)
#
#
# # Transformations to be applied to the training data
# T_trn = T.Compose([
#     T.Resize(32),  # Resize shortest size of the image to a fixed size.
#     # T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=30/360)], p=0.3), # Random color jittering
#     # T.RandomApply([T.ColorJitter(saturation=1.)], p=0.1), # Randomly transform image saturation from nothing to full grayscale
#     # T.RandomHorizontalFlip(), # Randomly flip image horizontally.
#     # T.RandomVerticalFlip(), # Randomly flip image vertically.
#     # T.RandomApply([T.Pad(16, padding_mode='reflect'), T.RandomRotation(10), T.CenterCrop(32)], p=0.3), # Random rotation
#     # T.RandomApply([T.RandomCrop(32, padding=8, pad_if_needed=True, padding_mode='reflect')], p=0.3), # Random translation and final cropping with fixed size
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
#
# # Transformations to be applied to the test data
# T_tst = T.Compose([
#     T.Resize(32),  # Resize shortest size of the image to a fixed size.
#     # T.CenterCrop(32), # Center crop of fixed size
#     T.ToTensor(),
#     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
#
#
# def get_data(dataset='cifar10', root='datasets', batch_size=32, num_workers=0, whiten_lvl=None):
#     trn_set, tst_set = None, None
#     if dataset == 'cifar10':
#         trn_set = CIFAR10(root=os.path.join(root, dataset), train=True, download=True, transform=T.ToTensor())
#         tst_set = CIFAR10(root=os.path.join(root, dataset), train=False, download=True, transform=T.ToTensor())
#     else:
#         raise NotImplementedError("Dataset {} not supported.".format(dataset))
#     zca = None
#     if whiten_lvl is not None:
#         print("Data whitening")
#         zca = ZCAWhitening()
#         all_data = np.concatenate([trn_set.data, tst_set.data], axis=0)
#         all_data = all_data.astype(np.float32) / 255.0  # Normalize data to [0, 1]
#         zca.fit(all_data)
#
#         trn_set.transform = T.Compose([
#             T.Resize(32),
#             T.ToTensor(),
#             ZCATransformation(zca),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#
#         tst_set.transform = T.Compose([
#             T.Resize(32),
#             T.ToTensor(),
#             ZCATransformation(zca),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#     else:
#
#
#     trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True, num_workers=P.NUM_WORKERS)
#     tst_loader = DataLoader(tst_set, batch_size=batch_size, shuffle=False, num_workers=P.NUM_WORKERS)
#
#     return trn_loader, tst_loader, None
#
#
# def whiten(inputs, zca):
#     return torch.tensor(zca.whiten(inputs.numpy().reshape(inputs.shape[0], -1)).reshape(inputs.shape))
