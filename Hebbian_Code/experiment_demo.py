from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched

import data
from model import Net
from model_triangle import Net_Triangle
from model_full import Net_Triangle

import utils
import numpy as np
import matplotlib.pyplot as plt
import umap
import warnings


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
                    print("Extracting model features")
                    features = model.features_extract(data)
                else:
                    print("Extracting conv features")
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
        plt.figure(figsize=(12, 10))
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

if __name__ == "__main__":
    hebb_param = {'mode': 'wta', 'w_nrm': False, 'bias': False, 'act': nn.Identity(), 'k': 1, 'alpha': 1.}
    device = torch.device('cuda:0')
    model = Net_Triangle(hebb_params=hebb_param)
    model.to(device)

    # unsup_optimizer = TensorLRSGD([
    #     {"params": model.conv1.parameters(), "lr": 0.08, },  # SGD does descent, so set lr to negative
    #     {"params": model.conv2.parameters(), "lr": 0.005, }
    # ], lr=0)
    hebb_params = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.conv4.parameters())
    unsup_optimizer = optim.SGD(hebb_params, lr=0.01)
    # unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)

    sup_optimizer = optim.Adam(model.fc1.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trn_set, tst_set, zca = data.get_data(dataset='cifar10', root='datasets', batch_size=32,
                                          whiten_lvl=1e-1)

    # Unsupervised training with SoftHebb
    running_loss = 0.0
    for epoch in range(5):

        for i, data in enumerate(trn_set, 0):
            inputs, _ = data
            inputs = inputs.to(device)
            # zero the parameter gradients
            unsup_optimizer.zero_grad()
            # forward + update computation
            with torch.no_grad():
                outputs = model(inputs)
            for layer in [model.conv1, model.conv2]:
                if hasattr(layer, 'local_update'):
                    layer.local_update()
            # optimize
            unsup_optimizer.step()
            # unsup_lr_scheduler.step()
        print("Visualizing Filters")
        model.visualize_filters('conv1', f'results/{"demo"}/demo_conv1_filters_epoch_{1}.png')
        model.visualize_filters('conv2', f'results/{"demo"}/demo_conv2_filters_epoch_{1}.png')


    # Supervised training of classifier
    # set requires grad false and eval mode for all modules but classifier
    print("Classifier")
    unsup_optimizer.zero_grad()
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.bn1.eval()
    model.bn2.eval()

    model.conv3.requires_grad = False
    model.conv4.requires_grad = False
    model.conv3.eval()
    model.conv4.eval()
    model.bn3.eval()
    model.bn4.eval()
    print("Visualizing Class separation")
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2)
    for epoch in range(50):
        model.fc1.train()
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