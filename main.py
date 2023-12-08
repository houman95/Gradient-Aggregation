import traceback
import copy
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
from multiprocessing import Pool
from functools import partial
import pickle
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
# Server Class
class Server:
    def __init__(self, clients, global_model, num_rounds, test_loader):
        self.clients = clients
        self.global_model = global_model
        self.num_rounds = num_rounds
        self.test_loader = test_loader

    def federated_learning(self, num_processes=8):
        all_rounds_info = {
            'rounds': [],
            'global_accuracies': [],
            'global_losses': [],
            'global_model_states': []
        }
        for round in range(self.num_rounds):
            #with Pool(processes=num_processes) as pool:
             #   model_differences = pool.map(partial(train_client, global_model=self.global_model), self.clients)

            model_differences = []
            for client in self.clients:
                globalcopy = copy.deepcopy(self.global_model)
                model_difference = train_client(client, globalcopy)
                model_differences.append(model_difference)

            # Aggregate and update global model
            avg_difference = self.average_model_differences(model_differences)
            self.global_model = self.reconstruct_model(self.global_model, avg_difference)
            # Evaluate and print global model accuracy
            global_accuracy,global_loss = evaluate_accuracy(self.global_model, self.test_loader)
            print(f"Round {round + 1}/{self.num_rounds}, Global Model Accuracy: {global_accuracy:.2f} , global model loss: {global_loss}%")
            # Save the results
            all_rounds_info['rounds'].append(round + 1)
            all_rounds_info['global_accuracies'].append(global_accuracy)
            all_rounds_info['global_losses'].append(global_loss)
            all_rounds_info['global_model_states'].append(copy.deepcopy(self.global_model.state_dict()))
        print(all_rounds_info['global_losses'])
        return all_rounds_info
    def average_model_differences(self, model_differences):
        avg_difference = {}
        for key in model_differences[0].keys():
            avg_difference[key] = sum(model_difference[key] for model_difference in model_differences) / len(
                model_differences)
        return avg_difference
    def reconstruct_model(self, original_model, model_difference):
        # Create a copy of the original model to keep it unchanged
        reconstructed_model = copy.deepcopy(original_model)
        device = next(reconstructed_model.parameters()).device  # Get the device from the model

        # Apply the differences to the reconstructed model's parameters
        with torch.no_grad():  # Ensure no gradient computation is done here
            for name, param in reconstructed_model.named_parameters():
                if name in model_difference:
                    # Move model_difference to the correct device
                    diff = model_difference[name].to(device)
                    # Update the parameter with the difference
                    param.data.add_(diff)

            # Update the running mean and variance for BatchNorm layers
            for name, module in reconstructed_model.named_modules():
                if isinstance(module, nn.BatchNorm2d):

                    if f"{name}.running_mean" in model_difference:
                        module.running_mean.copy_(model_difference[f"{name}.running_mean"])
                        running_mean_diff = model_difference[f"{name}.running_mean"].to(device)
                    if f"{name}.running_var" in model_difference:
                        running_var_diff = model_difference[f"{name}.running_var"].to(device)
                        module.running_var.copy_(running_var_diff)
        return reconstructed_model
def train_client(client, global_model):
    # Train the client using the global model and return the model difference
    return client.train(global_model)
# Client Class
class Client:
    def __init__(self, client_id, global_model, train_data, criterion, num_epochs):
        self.client_id = client_id
        self.train_data = train_data
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.global_model = global_model

    # Create the DataLoader for training data
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    def train(self, global_model):
        # Load the state dict into the local model
        # Create a local copy of the global model for training
        local_model = copy.deepcopy(global_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_model.to(device)
        print("Model device:", next(local_model.parameters()).device)

        optimizer = optim.SGD(local_model.parameters(), lr=0.001, momentum=0.9)

        # Train the local model
        trained_model, model_difference, _, _ = train_model(self.client_id, local_model, self.criterion, optimizer, self.train_loader,
                                                                self.num_epochs)
        return model_difference
def train_model(clientid, model, criterion, optimizer, train_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA Available:", torch.cuda.is_available())
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

        print(f"client {clientid}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Calculate the parameter differences post-training
    model_difference = {name: model.state_dict()[name] - pre_training_state[name]
                        for name in model.state_dict()}

    # Copy BatchNorm running mean and variance directly
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            model_difference[f"{name}.running_mean"] = model.state_dict()[f"{name}.running_mean"].clone()
            model_difference[f"{name}.running_var"] = model.state_dict()[f"{name}.running_var"].clone()
    return model, model_difference, epoch_losses, epoch_accuracies



#Function to evaluate the accuracy of a model
def evaluate_accuracy(model, data_loader):
    device = next(model.parameters()).device  # Get the device model is on
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
            outputs = model(images)
            loss = criterion(outputs,labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss



def main():
    try:
        #mp.set_start_method('spawn') commented
        #DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_epochs = 3
        num_rounds = 10
        num_clients = 1
        fraction_of_dataset = 1  # 20% of the dataset for each client

        # Load and prepare CIFAR10 dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_splits, test_dataset = load_cifar10_splits(num_clients)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

        # Initialize clients and global model
        clients = []
        global_model = models.resnet18(pretrained=False)
        global_model.fc = nn.Linear(global_model.fc.in_features, 10)
        global_model = global_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Training Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)
        # Load the CIFAR10 dataset and split it into training subsets for each client
        train_splits, test_dataset = load_cifar10_splits(num_clients)
        # Create Client objects, each with its own subset of the training data
        clients = [Client(i, global_model, train_splits[i], criterion, num_epochs) for i in range(num_clients)]

        # Perform federated learning
        server = Server(clients, global_model, num_rounds, test_loader)
        all_rounds_info = server.federated_learning()

        # Save the results
        with open('all_rounds_info.pkl', 'wb') as file:
            pickle.dump(all_rounds_info, file)


    except Exception as e:
        print(f"An error occured:{e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()