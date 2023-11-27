import traceback
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import pickle


class Client:
    def __init__(self, train_loader, criterion, num_epochs):
        self.train_loader = train_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
    def train(self, global_model):
        # Create a local copy of the global model for training
        local_model = copy.deepcopy(global_model)
        optimizer = optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)

        # Train the local model
        trained_model, model_difference, _, _ = train_model(local_model, self.criterion, optimizer, self.train_loader, self.num_epochs)
        return model_difference

def average_model_differences(model_differences):
    avg_difference = {}
    for key in model_differences[0].keys():
        avg_difference[key] = sum(model_difference[key] for model_difference in model_differences) / len(
            model_differences)
    return avg_difference
def federated_learning(clients, global_model, num_rounds, test_loader):
    all_rounds_info = []

    for round in range(num_rounds):
        model_differences = []

        # Each client trains and sends model difference
        for client in clients:
            model_difference = client.train(global_model)
            model_differences.append(model_difference)

        # Average the model differences
        averaged_difference = average_model_differences(model_differences)

        # Reconstruct the global model
        global_model = reconstruct_model(global_model, averaged_difference)

        # Evaluate global model accuracy
        global_accuracy = evaluate_accuracy(global_model, test_loader)
        all_rounds_info.append(global_accuracy)
        print(f"Round {round + 1}/{num_rounds}, Global Model Accuracy: {global_accuracy:.2f}%")

    return all_rounds_info

def train_model(model, criterion, optimizer, train_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Store the complete pre-training state of the model
    pre_training_state = copy.deepcopy(model.state_dict())
    epoch_losses = []
    epoch_accuracies = []

    # Training Loop

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Calculate the parameter differences post-training
    model_difference = {name: model.state_dict()[name] - pre_training_state[name]
                        for name in model.state_dict()}

    # Copy BatchNorm running mean and variance directly
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            model_difference[f"{name}.running_mean"] = model.state_dict()[f"{name}.running_mean"].clone()
            model_difference[f"{name}.running_var"] = model.state_dict()[f"{name}.running_var"].clone()

    return model, model_difference, epoch_losses, epoch_accuracies


def reconstruct_model(original_model, model_difference):
    # Create a copy of the original model to keep it unchanged
    reconstructed_model = copy.deepcopy(original_model)

    # Apply the differences to the reconstructed model's parameters
    with torch.no_grad():  # Ensure no gradient computation is done here
        for name, param in reconstructed_model.named_parameters():
            if name in model_difference:
                # Update the parameter with the difference
                param.data.add_(model_difference[name])

        # Update the running mean and variance for BatchNorm layers
        for name, module in reconstructed_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                if f"{name}.running_mean" in model_difference:
                    module.running_mean.copy_(model_difference[f"{name}.running_mean"])
                if f"{name}.running_var" in model_difference:
                    module.running_var.copy_(model_difference[f"{name}.running_var"])

    return reconstructed_model
def load_cifar10(root='./data', fraction=0.02, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # Reduce dataset size
    indices = torch.randperm(len(train_dataset)).tolist()
    reduced_size = int(fraction * len(indices))
    train_dataset_reduced = Subset(train_dataset, indices[:reduced_size])

    return train_dataset_reduced, test_dataset



# Function to evaluate the accuracy of a model
def evaluate_accuracy(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy



def main():
    try:
        num_epochs = 3
        num_rounds = 4
        num_clients = 2
        fraction_of_dataset = 0.02  # 2% of the dataset for each client

        # Load and prepare CIFAR10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set, test_set = load_cifar10(root='./data', fraction=fraction_of_dataset, transform=transform)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

        # Initialize clients and global model
        clients = []
        global_model = models.resnet18(pretrained=False)
        global_model.fc = nn.Linear(global_model.fc.in_features, 10)
        # Training Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)

        for i in range(num_clients):
            client_train_set = Subset(train_set, torch.randperm(len(train_set)).tolist()[:int(fraction_of_dataset * len(train_set))])
            client_train_loader = DataLoader(client_train_set, batch_size=64, shuffle=True, num_workers=2)
            client = Client(client_train_loader, criterion, num_epochs)
            clients.append(client)
        # Perform federated learning
        all_rounds_info = federated_learning(clients, global_model, num_rounds, test_loader)

        # Save the results
        with open('all_rounds_info.pkl', 'wb') as file:
            pickle.dump(all_rounds_info, file)
    except Exception as e:
        print(f"An error occured:{e}")
        traceback.print_exc()
    return all_rounds_info
if __name__ == "__main__":
    all_rounds_info = main()
    with open('all_rounds_info.pkl', 'wb') as file:
        pickle.dump(all_rounds_info, file)
