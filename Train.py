import torch
import torch.nn as nn
import torch.nn.functional as F
from HebbianClassifier import HebbianClassifier
from HebbianConv import HebbianConv2d



class HebbianCNN(nn.Module):
    def __init__(self, in_channels, conv_out_channels, kernel_size, num_classes,
                 conv_mode='wta', conv_alpha=1.0, conv_lr=0.01,
                 classifier_mode='oja', classifier_lr=0.01):
        super().__init__()

        self.conv = HebbianConv2d(in_channels, conv_out_channels, kernel_size,
                                  mode=conv_mode, alpha=conv_alpha, lr=conv_lr)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = HebbianClassifier(conv_out_channels, num_classes,
                                            lr=classifier_lr, mode=classifier_mode)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x


class HebbianTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def train_step(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)

        # Forward pass
        output = self.model(x)

        # Compute loss (for monitoring purposes only)
        loss = F.cross_entropy(output, y)

        # Apply Hebbian updates
        self.model.conv.local_update()
        self.model.classifier.update(self.model.gap(self.model.conv(x)).view(x.size(0), -1), output)
        self.model.classifier.apply_update()

        return loss.item()

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for x, y in dataloader:
                loss = self.train_step(x, y)
                total_loss += loss
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total


# Example usage
in_channels = 3
conv_out_channels = 64
kernel_size = 3
num_classes = 10

model = HebbianCNN(in_channels, conv_out_channels, kernel_size, num_classes)
trainer = HebbianTrainer(model)

# Create dummy dataloaders
train_dataloader = [(torch.randn(32, in_channels, 32, 32), torch.randint(0, num_classes, (32,))) for _ in range(100)]
test_dataloader = [(torch.randn(32, in_channels, 32, 32), torch.randint(0, num_classes, (32,))) for _ in range(20)]

# Train the model
trainer.train(train_dataloader, num_epochs=5)

# Evaluate the model
accuracy = trainer.evaluate(test_dataloader)
print(f"Test Accuracy: {accuracy:.2f}")