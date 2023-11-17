import traceback
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # MNIST images are grayscale, so 1 input channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted for MNIST image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 output classes (digits 0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_mnist(root='./data',
               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])):
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def train_model(model, criterion, optimizer, train_loader, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Store the pre-training state
    pre_training_state = {name: param.clone() for name, param in model.named_parameters()}

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Calculate the parameter differences
    model_difference = {name: model.state_dict()[name] - pre_training_state[name] for name in model.state_dict()}

    return model, model_difference



def reconstruct_model(original_model, model_difference):
    # Create a copy of the original model to keep it unchanged
    reconstructed_model = copy.deepcopy(original_model)

    # Apply the differences to the reconstructed model's parameters
    with torch.no_grad():  # Ensure no gradient computation is done here
        for name, param in reconstructed_model.named_parameters():
            if name in model_difference:
                # Update the parameter with the difference
                param.data.add_(model_difference[name])

    return reconstructed_model


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

def compare_models(trained_model, reconstructed_model):
    for (name1, param1), (name2, param2) in zip(trained_model.named_parameters(), reconstructed_model.named_parameters()):
        if param1.data.nelement() == 0:  # Skip parameters with no data
            continue

        # Calculate the relative difference
        relative_diff = torch.abs(np.subtract(param1, param2)) / torch.abs(param1)
        counter = 0
        # Check if any element in the difference tensor exceeds the threshold (1%)
        if torch.any(relative_diff > 0.01):
            counter += 1
            print(torch.norm(param1 - param2))
            print(f"Parameter: {name1}")
            print(f"Trained Model: {param1.data}")
            print(f"Reconstructed Model: {param2.data}\n")
    print(counter)


def main():
    try:
        num_epochs = 5
        # Step 1: Load and Prepare the CIFAR10 Dataset or MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize for 1-channel grayscale images
        ])

        train_set, test_set = load_mnist(root='./data', transform=transform)

        indices = torch.randperm(len(train_set)).tolist()
        train_set_reduced = Subset(train_set, indices[:int(0.1 * len(indices))])

        train_loader = DataLoader(train_set_reduced, batch_size=64, shuffle=True, num_workers=2)

        # Step 2: Define the ResNet18 Model or LeNet
        #model = models.resnet18(pretrained=False)
        #model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
        model = LeNet()
        model_copy = copy.deepcopy(model)

        # Step 3: Training Loop
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        trained_model, model_difference = train_model(model, criterion, optimizer, train_loader, num_epochs)

        # Step 5: Reconstruct the Model
        reconstructed_model = reconstruct_model(model_copy, model_difference)

        # Step 6: Evaluate the Model
        # Here you would include code to evaluate the accuracy of trained_model and reconstructed_model
        # Evaluate accuracies
        # Step 1: Load the CIFAR10 Test Dataset
        #test_transform = transforms.Compose([transforms.ToTensor(),
                                             #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

        trained_model_accuracy = evaluate_accuracy(trained_model, test_loader)
        reconstructed_model_accuracy = evaluate_accuracy(reconstructed_model, test_loader)

        print(f"Accuracy of Trained Model: {trained_model_accuracy:.2f}%")
        print(f"Accuracy of Reconstructed Model: {reconstructed_model_accuracy:.2f}%")

        # Call this function in your main after reconstructing the model
        compare_models(trained_model, reconstructed_model)
    except Exception as e:
        print(f"An error occured : {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()