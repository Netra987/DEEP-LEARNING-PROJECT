import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    # === 1. Device Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 2. Dataset and Dataloader ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))  # Data augmentation
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                ]))

    # Set num_workers=0 for Windows compatibility
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)

    # === 3. CNN Model ===
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.25)
            self.fc1 = nn.Linear(64*7*7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.dropout(x)
            x = x.view(-1, 64*7*7)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # === 4. Training ===
    def train(model, device, train_loader, optimizer, criterion, epoch):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        accuracy = 100. * correct / total
        avg_loss = train_loss / len(train_loader)
        print(f'Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')
        return avg_loss, accuracy

    def test(model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
        return avg_loss, accuracy, all_preds, all_targets

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    for epoch in range(1, 11):  # Increased to 10 epochs
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc, all_preds, all_targets = test(model, device, test_loader, criterion)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step(test_loss)

    # === 5. Visualization ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # === 6. Prediction Visualization ===
    def plot_predictions(images, labels, predictions, num_images=5):
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for idx in range(num_images):
            ax = axes[idx]
            ax.imshow(images[idx][0].cpu(), cmap='gray')
            ax.set_title(f'True: {labels[idx]}\nPred: {predictions[idx]}', 
                        color='green' if labels[idx] == predictions[idx] else 'red')
            ax.axis('off')
        plt.show()

    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)
    example_data = example_data.to(device)

    with torch.no_grad():
        output = model(example_data)
        _, preds = torch.max(output, 1)

    plot_predictions(example_data[:5].cpu(), example_targets[:5].numpy(), preds[:5].cpu().numpy())

    # Save the model
    torch.save(model.state_dict(), 'mnist_cnn.pth')

if __name__ == '__main__':
    main()
