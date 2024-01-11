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
import random
import logging

# Configure logging
logging.basicConfig(filename='federated_learningsimp.log', level=logging.INFO, filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_last_round_model_params(file_name):
    with open(file_name, 'rb') as file:
        all_rounds_info = pickle.load(file)
    last_round_params = all_rounds_info['global_model_states'][-1]
    return last_round_params


def decide_transmission(num_clients):
    return [random.random() < 1.0 / num_clients for _ in range(num_clients)]


def initialize_global_model(model, model_params):
    model.load_state_dict(model_params)
    return model




def load_cifar10_splits(num_splits, root='./data', fraction=1, shuffle=True):
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
    test_dataset = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=transform_test)

    # Shuffle the training dataset
    indices = torch.randperm(len(train_dataset)).tolist() if shuffle else list(range(len(train_dataset)))
    split_size = int(fraction * len(indices) / num_splits)
    train_splits = []
    for i in range(num_splits):
        split_indices = indices[i * split_size:(i + 1) * split_size]
        train_splits.append(Subset(train_dataset, split_indices))

    return train_splits, test_dataset

# Server Class
class Server:
    def __init__(self, clients, global_model, num_rounds, test_loader, framelength):
        self.clients = clients
        self.global_model = global_model
        self.num_rounds = num_rounds
        self.test_loader = test_loader
        self.framelength = framelength
    def update_client_models(self):
        global_model_params = self.global_model.state_dict()
        for client in self.clients:
            client.local_model.load_state_dict(global_model_params)
    def federated_learning(self, num_processes=8):
        all_rounds_info = {
            'rounds': [],
            'global_accuracies': [],
            'global_losses': [],
            'global_model_states': []
        }
        for round in range(self.num_rounds):
            successful_clients = set()
            # Framed Slotted ALOHA mechanism
            for slot in range(self.framelength):
                transmission_decisions = decide_transmission(len(self.clients))
                if transmission_decisions.count(True) == 1:  # Only one client transmitted successfully
                    successful_client_index = transmission_decisions.index(True)
                    successful_clients.add(successful_client_index)
            # Handle no successful transmission
            if not successful_clients:
                logging.info(
                    f"No successful transmission in round {round + 1}. Training all clients without updating the server model.")
                for client in self.clients:
                    client.local_model.load_state_dict(client.trained_model_state)
                    _ = train_client(client)  # Train client but do not collect model difference
                continue  # Skip model update and proceed to the next round 
            model_differences = []
            if successful_clients:                
                for client_index in successful_clients:
                    client = self.clients[client_index]
                    client.local_model.load_state_dict(client.trained_model_state)
                    model_difference = train_client(client)
                    model_differences.append(model_difference)

            # Aggregate and update global model
            avg_difference = self.average_model_differences(model_differences)
            self.global_model = self.reconstruct_model(self.global_model, avg_difference)
            for client in self.clients:
                client.trained_model_state = copy.deepcopy(self.global_model.state_dict())
                client.local_model.load_state_dict(client.trained_model_state)
            # Evaluate and print global model accuracy
            global_accuracy, global_loss = evaluate_accuracy(self.global_model, self.test_loader)
            logging.info(
                f"Round {round + 1}/{self.num_rounds}, Global Model Accuracy: {global_accuracy:.2f} , global model loss: {global_loss}%")
            # Save the results
            all_rounds_info['rounds'].append(round + 1)
            all_rounds_info['global_accuracies'].append(global_accuracy)
            all_rounds_info['global_losses'].append(global_loss)
            #all_rounds_info['global_model_states'].append(copy.deepcopy(self.global_model.state_dict()))
            if round>0:
                train_splits, _ = load_cifar10_splits(len(self.clients))
                for i, client in enumerate(self.clients):
                        client.train_data = train_splits[i]
                        if i == 0:
                            first_five_ids = client.train_data.indices[:5]
                            logging.info(f"Client 0 - First 5 Data Point IDs after reshuffling: {first_five_ids}")

        print(all_rounds_info['global_losses'])
        if not (round % 5):
            # Log memory info before clearing cache
            allocated_before = torch.cuda.memory_allocated()
            reserved_before = torch.cuda.memory_reserved()
            logging.info(f"CUDA memory before clearing cache: Allocated: {allocated_before}, Reserved: {reserved_before}")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            # Log memory info after clearing cache
            allocated_after = torch.cuda.memory_allocated()
            reserved_after = torch.cuda.memory_reserved()
            logging.info(f"CUDA memory after clearing cache: Allocated: {allocated_after}, Reserved: {reserved_after}%")
        
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


def train_client(client):
    # Train the client using the global model and return the model difference
    return client.train()


# Client Class
class Client:
    def __init__(self, client_id, global_model, train_data, test_data, criterion, num_epochs):
        self.client_id = client_id
        self.train_data = train_data
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.local_model = copy.deepcopy(global_model)
        self.test_loader = test_data
        # Create the DataLoader for training data
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
        self.trained_model_state = copy.deepcopy(global_model.state_dict())

    def train(self):
        self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=0)
	    # Log information about the new dataset
        #sample_indices = [self.train_data.indices[0], self.train_data.indices[1], self.train_data.indices[2], self.train_data.indices[3], self.train_data.indices[4]]
        #logging.info(f"Client {self.client_id} training on data: {sample_indices[:5]}")

        # Load the trained model state from the previous round if available
        if hasattr(self, 'trained_model_state'):
            self.local_model.load_state_dict(self.trained_model_state)
        # Load the state dict into the local model
        # Create a local copy of the global model for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_model.to(device)
        #logging.info("Model device:", next(self.local_model.parameters()).device)

        optimizer = optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.9)

        # Train the local model
        self.local_model, model_difference, _, _ = train_model(self.client_id, self.local_model, self.criterion,
                                                               optimizer, self.train_loader,
                                                               self.num_epochs, self.test_loader)
		# Save the trained model state for use in the next round
        self.trained_model_state = copy.deepcopy(self.local_model.state_dict())
        return model_difference


def train_model(clientid, model, criterion, optimizer, train_loader, num_epochs,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #logging.info("CUDA Available:", torch.cuda.is_available())
    model = model.to(device)
    # Evaluate model accuracy before training
    # Evaluate model accuracy and loss before training
    pre_training_accuracy, pre_training_loss = evaluate_accuracy(model, test_loader)
    logging.info(f"Client {clientid}, Pre-training Accuracy: {pre_training_accuracy:.2f}%, Pre-training Loss: {pre_training_loss:.2f}")
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

        logging.info(f"client {clientid}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Calculate the parameter differences post-training
    model_difference = {name: model.state_dict()[name] - pre_training_state[name]
                        for name in model.state_dict()}

    # Copy BatchNorm running mean and variance directly
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            model_difference[f"{name}.running_mean"] = model.state_dict()[f"{name}.running_mean"].clone()
            model_difference[f"{name}.running_var"] = model.state_dict()[f"{name}.running_var"].clone()
    return model, model_difference, epoch_losses, epoch_accuracies


# Function to evaluate the accuracy of a model
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
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss


def main():
    try:
        # mp.set_start_method('spawn')
        # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_epochs = 4
        num_rounds = 40
        num_clients = 100
        frame_length = 30
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
        # Load parameters from last round
        #last_round_params = load_last_round_model_params('all_rounds_info100clients3epochfrwd50.pkl')
        global_model = models.resnet18(pretrained=False)
        global_model.fc = nn.Linear(global_model.fc.in_features, 10)
        #global_model = initialize_global_model(global_model, last_round_params)
        global_model = global_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Training Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(global_model.parameters(), lr=0.001, momentum=0.9)
        # Load the CIFAR10 dataset and split it into training subsets for each client
        train_splits, test_dataset = load_cifar10_splits(num_clients)
        # Create Client objects, each with its own subset of the training data
        clients = [Client(i, global_model, train_splits[i], test_loader, criterion, num_epochs) for i in range(num_clients)]

        # Perform federated learning
        server = Server(clients, global_model, num_rounds, test_loader, framelength=frame_length)
        all_rounds_info = server.federated_learning()
        filename = f'all_rounds_info{num_clients}clients{frame_length}framewidth{num_epochs}epoch.pkl'

        # Save the results
        with open(filename, 'wb') as file:
            pickle.dump(all_rounds_info, file)

    except Exception as e:
        logging.error(f"An error occured:{e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()