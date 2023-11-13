import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST, CIFAR10
from torch.optim import SGD
from functools import partial


############################################################
###################### Nodel and Data ######################
############################################################
# Define the ResNet-18 model
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


# Update the dataset loading
def load_cifar10_splits(num_splits, root='./data', fraction=0.5):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    train_dataset, _ = random_split(train_dataset, [int(fraction * len(train_dataset)),
                                                    len(train_dataset) - int(fraction * len(train_dataset))])

    test_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    # Split the training dataset
    train_splits = random_split(train_dataset, [len(train_dataset) // num_splits] * num_splits)

    return train_splits, test_dataset


######################################################################
###################### Train and Test Functions ######################
######################################################################
def train_fn(model, train_loader, epochs=1, optimizer=partial(SGD, lr=0.01, momentum=0), criterion=CrossEntropyLoss(),
             verbose=True):
    device = next(model.parameters()).device
    optimizer = optimizer(model.parameters())
    model.train()
    running_loss = 0.0
    gradients = []  # Initialize a list to store gradients
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / (epochs * len(train_loader))  # , gradients # Return the loss and gradients


def test_fn(model, test_loader, criterion=CrossEntropyLoss(), verbose=True):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_loss /= len(test_loader)

    return accuracy, test_loss


##################################################
###################### Data ######################
##################################################


def cifar10_iid_split(dataset, num_splits):
    # Splits a PyTorch Dataset into N IID datasets.

    # Args:
    #    dataset (torch.utils.data.Dataset): The dataset to split.
    #    num_splits (int): The number of IID splits to create.

    # Returns:
    #    list: A list of datasets, each an IID split of the original.

    assert num_splits > 0, "Number of splits must be a positive integer."

    # Compute the length of each split
    split_length = len(dataset) // num_splits

    # Compute the lengths of the splits
    lengths = [split_length] * num_splits

    # Adjust the first split length if the dataset is not evenly divisible by num_splits
    lengths[0] += len(dataset) - sum(lengths)

    # Randomly split the dataset
    splits = random_split(dataset, lengths)

    return splits
