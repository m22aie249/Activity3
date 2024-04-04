import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

print("Adding debug to roll fwd")

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the STL10 dataset
train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=data_transforms['train'])
test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=data_transforms['test'])

class SubsampledDataset(Dataset):
    def __init__(self, dataset, subsample_size):
        self.dataset = dataset
        self.subsample_size = subsample_size
        self.indices = torch.randperm(len(dataset))[:subsample_size]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.subsample_size

subsample_size = 500
subsample_test_size = 100

sub_sampled_train_dataset = SubsampledDataset(train_dataset, subsample_size)
sub_sampled_test_dataset = SubsampledDataset(test_dataset, subsample_test_size)

# Define the dataloaders
train_loader = torch.utils.data.DataLoader(sub_sampled_train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(sub_sampled_test_dataset, batch_size=64, shuffle=False)

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

# Load the pre-trained ResNet101 model
model = models.resnet101(pretrained=True)

# Freeze all layers except the final layer
freeze_layers(model)

# Modify the final layer
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 20) 

# V2 change: Add dropout for regularization
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),            
    nn.Linear(512, 10)
)

# Move the model to the device
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# List of optimizers to use
optimizers = [optim.Adam, optim.Adagrad, optim.Adadelta, optim.RMSprop]
optimizer_names = ["Adam", "Adagrad", "Adadelta", "RMSprop"]

def train_model(model, criterion, optimizer, num_epochs=1):
    model.train()
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(sub_sampled_train_dataset)
        epoch_acc = running_corrects.double() / len(sub_sampled_train_dataset)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        print('Epoch {}/{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, num_epochs - 1, epoch_loss, epoch_acc))

    return model, losses, accuracies

def plot_curves(losses, accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.show()

def top5_accuracy(outputs, labels):
    _, top5_preds = torch.topk(outputs, 5, dim=1)
    correct = top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds))
    return correct.float().sum().item() / labels.size(0)

for optimizer_name, optimizer_func in zip(optimizer_names, optimizers):
    print(f"Training with {optimizer_name} optimizer...")
    optimizer = optimizer_func(model.parameters(), lr=0.001)
    model, losses, accuracies = train_model(model, criterion, optimizer)
    plot_curves(losses, accuracies)

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels.data).sum().item()

    test_loss /= len(sub_sampled_test_dataset)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} Test Accuracy: {test_acc:.4f}")

    # Calculate and report the final top-5 test accuracy
    top5_test_acc = top5_accuracy(outputs, labels)
    print(f"Top-5 Test Accuracy: {top5_test_acc:.4f}\n")