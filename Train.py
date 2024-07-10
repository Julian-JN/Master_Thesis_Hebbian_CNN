import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from HebbianConv import HebbianConv2d
from HebbianClassifier import LinearHebbianClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class HebbianCNN(nn.Module):
    def __init__(self, num_classes, learning_rate=0.01, mode='hebbian', wta_competition='filter'):
        super(HebbianCNN, self).__init__()
        self.conv1 = HebbianConv2d(in_channels=1, out_channels=16, kernel_size=3, learning_rate=learning_rate,
                                   mode=mode, wta_competition=wta_competition)
        self.conv2 = HebbianConv2d(in_channels=16, out_channels=32, kernel_size=3, learning_rate=learning_rate,
                                   mode=mode, wta_competition=wta_competition)
        self.fc1 = LinearHebbianClassifier(input_dim=32 * 5 * 5, output_dim=128, learning_rate=learning_rate, mode=mode,
                                           wta_competition=wta_competition)
        self.fc2 = LinearHebbianClassifier(input_dim=128, output_dim=num_classes, learning_rate=learning_rate,
                                           mode=mode, wta_competition=wta_competition)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def update_weights(self, x):
        conv1_output = self.conv1(x)
        self.conv1.update_weights(x, conv1_output)

        conv1_activated = F.relu(F.max_pool2d(conv1_output, 2))
        conv2_output = self.conv2(conv1_activated)
        self.conv2.update_weights(conv1_activated, conv2_output)

        conv2_activated = F.relu(F.max_pool2d(conv2_output, 2))
        x_flatten = conv2_activated.view(x.size(0), -1)
        fc1_output = F.relu(self.fc1(x_flatten))
        self.fc1.update_weights(x_flatten, fc1_output)

        fc2_output = self.fc2(fc1_output)
        self.fc2.update_weights(fc1_output, fc2_output)

        return fc2_output


def train(model, device, train_loader, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model.update_weights(data)

        if batch_idx % 100 == 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tAccuracy: {accuracy:.2f}%')


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    model = HebbianCNN(num_classes=10).to(device)

    for epoch in range(1, 11):
        train(model, device, train_loader, epoch)


if __name__ == '__main__':
    main()

