import argparse
from time import time
import os
import shutil
import copy

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.utils.tensorboard import SummaryWriter

import data
from model import Net
from model_triangle import Net_Triangle
import utils
import params as P
import numpy as np
import matplotlib.pyplot as plt
import umap
import warnings


def hebbian_train_one_epoch(model, train_loader, device, zca):
    for inputs, _ in tqdm(train_loader, ncols=80):
        inputs = inputs.to(device)
        # if zca is not None:
        #     inputs = data.whiten(inputs, zca)
        with torch.no_grad():
            outputs = model(inputs)  # Forward pass through the entire network
        for layer in [model.conv1, model.conv2]:
            if hasattr(layer, 'local_update'):
                layer.local_update()


def train_one_epoch(model, criterion, optimizer, train_loader, device, zca, tboard, epoch):
    epoch_loss, epoch_hits, count = 0, 0, 0
    grads = {}
    for inputs, labels in tqdm(train_loader, ncols=80):
        inputs, labels = inputs.to(device), labels.to(device)
        # if zca is not None:
        #     inputs = data.whiten(inputs, zca)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.sum().item()
        epoch_hits += (torch.max(outputs, dim=1)[1] == labels).int().sum().item()
        count += labels.shape[0]

        for n, p in model.named_parameters():
            if p.grad is None:
                continue
            grad = p.grad.clone().detach()
            if n not in grads:
                grads[n] = 0
            grads[n] = grads[n] + grad

    trn_loss, trn_acc = epoch_loss / count, epoch_hits / count
    tboard.add_scalar("Loss/train", trn_loss, epoch)
    tboard.add_scalar("Accuracy/train", trn_acc, epoch)
    return trn_loss, trn_acc, grads


def test_one_epoch(model, criterion, test_loader, device, zca, tboard, epoch):
    epoch_loss, epoch_hits, count = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, ncols=80):
            inputs, labels = inputs.to(device), labels.to(device)
            # if zca is not None:
            #     inputs = data.whiten(inputs, zca)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.sum().item()
            epoch_hits += (torch.max(outputs, dim=1)[1] == labels).int().sum().item()
            count += labels.shape[0]

    tst_loss, tst_acc = epoch_loss / count, epoch_hits / count
    tboard.add_scalar("Loss/test", tst_loss, epoch)
    tboard.add_scalar("Accuracy/test", tst_acc, epoch)
    return tst_loss, tst_acc


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
                if hasattr(model, 'forward_features'):
                    print("Extracting model features")
                    features = model.forward_features(data)
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

def run(exp_name, dataset='cifar10', whiten_lvl=None, batch_size=32, epochs=20,
        lr=1e-3, momentum=0.9, wdecay=0., sched_milestones=(), sched_gamma=1., hebb_params=None):
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=P.DEFAULT_DEVICE, choices=P.AVAILABLE_DEVICES,
                        help="The device you want to use for the experiment.")
    args = parser.parse_args()

    device = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    trn_set, tst_set, zca = data.get_data(dataset=dataset, root='datasets', batch_size=batch_size,
                                          whiten_lvl=whiten_lvl)

    # print("Visualising Clusters in Test Set!")
    # visualize_data_clusters(tst_set, method='umap', dim=3)
    model = Net_Triangle(hebb_params)
    model.to(device=device)

    # Hebbian training
    print("Starting Hebbian training...")
    # Optimizer only for the Hebbian layers
    # hebb_params = list(model.conv1.parameters()) + list(model.conv2.parameters()) + list(model.conv3.parameters()) + list(model.conv4.parameters())
    unsup_optimizer = TensorLRSGD([
        {"params": model.conv1.parameters(), "lr": 0.08, },  # SGD does descent, so set lr to negative
        {"params": model.conv2.parameters(), "lr": 0.005, }
    ], lr=0)

    unsup_lr_scheduler = WeightNormDependentLR(unsup_optimizer, power_lr=0.5)
    # model.visualize_in_input_space(tst_set, num_batches=10)
    for epoch in range(1):
        unsup_optimizer.zero_grad()
        hebbian_train_one_epoch(model, trn_set, device, zca)
        unsup_optimizer.step()
        unsup_lr_scheduler.step()
        print(f"Completed Hebbian training epoch {epoch + 1}/{5}")

    unsup_optimizer.zero_grad()
    print("Visualizing Filters")
    model.visualize_filters('conv1', f'results/{exp_name}/conv1_filters_epoch_{epoch}.png')
    model.visualize_filters('conv2', f'results/{exp_name}/conv2_filters_epoch_{epoch}.png')
    # print("Visualizing Weight to Data")
    # model.visualize_in_input_space(tst_set)

    print("Visualizing Class separation")
    visualize_data_clusters(tst_set, model=model, method='umap', dim=2)

    # Freeze Hebbian layers
    model.conv1.requires_grad = False
    model.conv2.requires_grad = False
    model.conv1.eval()
    model.conv2.eval()
    model.bn1.eval()
    model.bn2.eval()
    criterion = nn.CrossEntropyLoss()
    # Should only Train Classifier
    class_params = list(model.fc1.parameters())
    optimizer = optim.Adam(class_params, lr=lr)
    # Can train whole modelm
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wdecay, nesterov=True)
    scheduler = sched.MultiStepLR(optimizer, milestones=sched_milestones, gamma=sched_gamma)

    results = {'trn_loss': {}, 'trn_acc': {}, 'tst_loss': {}, 'tst_acc': {}}
    weight_stats, weight_update_stats, grad_stats = {}, {}, {}
    weight_dist, weight_update_dist, grad_dist = {}, {}, {}
    best_epoch, best_result = None, None
    if os.path.exists('tboard/{}'.format(exp_name)): shutil.rmtree('tboard/{}'.format(exp_name))
    tboard = SummaryWriter('tboard/{}'.format(exp_name))

    print("Training Classifier")
    for epoch in range(1, epochs + 1):
        t0 = time()
        print("\nEPOCH {}/{} | {}".format(epoch, epochs, exp_name))
        # Training phase
        model.fc1.train()
        model.dropout.train()
        # model.fc2.train()
        weights, weight_updates, grads = {n: copy.deepcopy(p) for n, p in model.named_parameters()}, {}, {}
        trn_loss, trn_acc, grads = train_one_epoch(model, criterion, optimizer, trn_set, device, zca, tboard, epoch)
        tboard.add_scalar("Loss/train", trn_loss, epoch)
        tboard.add_scalar("Accuracy/train", trn_acc, epoch)
        results['trn_loss'][epoch], results['trn_acc'][epoch] = trn_loss, trn_acc
        print("Train loss: {}, accuracy: {}".format(trn_loss, trn_acc))

        # Track weight, weight update, and gradient stats
        for n, p in model.named_parameters(): weight_updates[n] = p - weights[n]
        weight_stats = utils.update_param_stats(weight_stats, {n: p for n, p in model.named_parameters()})
        weight_dist = utils.update_param_dist(weight_dist, {n: p for n, p in model.named_parameters()})
        for n, s in weight_stats.items(): tboard.add_scalar("Weight/{}".format(n), s[-1], epoch)
        weight_update_stats = utils.update_param_stats(weight_update_stats, weight_updates)
        weight_update_dist = utils.update_param_dist(weight_update_dist, weight_updates)
        for n, s in weight_update_stats.items(): tboard.add_scalar("Delta_W/{}".format(n), s[-1], epoch)
        grad_stats = utils.update_param_stats(grad_stats, grads)
        grad_dist = utils.update_param_dist(grad_dist, grads)
        for n, s in grad_stats.items(): tboard.add_scalar("Grad/{}".format(n), s[-1], epoch)

        # Testing phase
        model.eval()
        print("Testing...")
        tst_loss, tst_acc = test_one_epoch(model, criterion, tst_set, device, zca, tboard, epoch)
        results['tst_loss'][epoch], results['tst_acc'][epoch] = tst_loss, tst_acc
        print("Test loss: {}, accuracy: {}".format(tst_loss, tst_acc))
        # Visualization
        # print("Visualizing Filters")
        # model.visualize_filters('conv1', f'results/{exp_name}/conv1_filters_epoch_{epoch}.png')
        # model.visualize_filters('conv2', f'results/{exp_name}/conv2_filters_epoch_{epoch}.png')
        tboard.add_scalar("Loss/test", trn_loss, epoch)
        tboard.add_scalar("Accuracy/test", trn_acc, epoch)

        # Keep track of best model
        print("Best model so far at epoch: {}, with result: {}".format(best_epoch, best_result))
        if best_result is None or best_result < tst_acc:
            print("New best model found!, Updating best model...")
            best_epoch = epoch
            best_result = tst_acc
            # utils.save_dict(copy.deepcopy(model).state_dict(), 'results/{}/best.pt'.format(exp_name))

        # Save results
        # print("Saving results...")
        # utils.update_csv(results, 'results/{}/results.csv'.format(exp_name))
        # utils.update_csv(weight_stats, 'results/{}/weight_stats.csv'.format(exp_name))
        # utils.update_csv(weight_update_stats, 'results/{}/weight_update_stats.csv'.format(exp_name))
        # utils.update_csv(grad_stats, 'results/{}/grad_stats.csv'.format(exp_name))
        # utils.update_csv(weight_dist, 'results/{}/weight_dist.csv'.format(exp_name))
        # utils.update_csv(weight_dist, 'results/{}/weight_update_dist.csv'.format(exp_name))
        # utils.update_csv(grad_dist, 'results/{}/grad_dist.csv'.format(exp_name))
        # print("Saving plots")
        # utils.save_plot({"Train": results['trn_loss'], "Test": results['tst_loss']},
        #                 'results/{}/figures/loss.png'.format(exp_name), xlabel="Epoch", ylabel="Loss")
        # utils.save_plot({"Train": results['trn_acc'], "Test": results['tst_acc']},
        #                 'results/{}/figures/accuracy.png'.format(exp_name), xlabel="Epoch", ylabel="Accuracy")
        # utils.save_grid_plot(weight_stats, 'results/{}/figures/weight_stats.png'.format(exp_name), rows=2,
        #                      cols=(len(weight_stats) + 1) // 2, ylabel="Weight Value")
        # utils.save_grid_plot(weight_update_stats, 'results/{}/figures/weight_update_stats.png'.format(exp_name), rows=2,
        #                      cols=(len(weight_update_stats) + 1) // 2, ylabel="Weight Update")
        # utils.save_grid_plot(grad_stats, 'results/{}/figures/grad_stats.png'.format(exp_name), rows=2,
        #                      cols=(len(grad_stats) + 1) // 2, ylabel="Grad. Value")
        # utils.save_grid_dist(weight_dist, 'results/{}/figures/weight_dist.png'.format(exp_name), rows=2,
        #                      cols=(len(weight_dist) + 1) // 2, bins=P.DIST_BINS)
        # utils.save_grid_dist(weight_update_dist, 'results/{}/figures/weight_update_dist.png'.format(exp_name), rows=2,
        #                      cols=(len(weight_update_dist) + 1) // 2, bins=P.DIST_BINS)
        # utils.save_grid_dist(grad_dist, 'results/{}/figures/grad_dist.png'.format(exp_name), rows=2,
        #                      cols=(len(grad_dist) + 1) // 2, bins=P.DIST_BINS)
        tboard.flush()
        # utils.save_dict(model.state_dict(), 'results/{}/last.pt'.format(exp_name))

        # LR scheduling
        scheduler.step()

        t = time() - t0
        print("Epoch duration: {}".format(utils.format_time(t)))
        print("Expected remaining time: {}".format(utils.format_time((epochs - epoch) * t)))

    tboard.close()

    print("\nFinished!")