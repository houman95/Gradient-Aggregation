import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def train_model(model, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        pre_training_state_dict = {key: value.clone() for key, value in model.state_dict().items()}

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        post_training_state_dict = model.state_dict()

        # Step 4: Compute Parameter Difference and Step 5: Reconstruct the Model
        for key in pre_training_state_dict:
            pre_training_state_dict[key] += post_training_state_dict[key] - pre_training_state_dict[key]

        model.load_state_dict(pre_training_state_dict)

    return model



# Define a function to evaluate the model's accuracy
def evaluate_model(model, data_loader):
    correct = 0
    total = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Evaluate the trained model
trained_model_accuracy = evaluate_model(trained_model, test_loader)

# If you have a reconstructed model (e.g., reconstructed_model), evaluate it as well
reconstructed_model_accuracy = evaluate_model(reconstructed_model, test_loader)

# Print the accuracy
print(f"Accuracy of the trained model: {trained_model_accuracy:.2f}%")
print(f"Accuracy of the reconstructed model: {reconstructed_model_accuracy:.2f}%")
def main():
    # [Load data, initialize model, criterion, optimizer]
    # Step 1: Load and Prepare the CIFAR10 Dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(train_set)).tolist()
    train_set_reduced = Subset(train_set, indices[:int(0.01 * len(indices))])

    train_loader = DataLoader(train_set_reduced, batch_size=4, shuffle=True, num_workers=2)

    # Step 2: Define the ResNet18 Model
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes

    # Step 3: Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # [other model setup]
    trained_model = train_model(model, criterion, optimizer, num_epochs=5, train_loader=train_loader)
    # [evaluation and other logic]
    # Step 6: Evaluate the Model
    # Here you would include code to evaluate the accuracy of trained_model
    # You can use a validation set or test set for this purpose
    # Step 6: Evaluate the Model

    # Load the CIFAR10 test dataset
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)
    trained_model_accuracy = evaluate_model(trained_model, test_loader)
    reconstructed_model_accuracy = evaluate_model(reconstructed_model, test_loader)

if __name__ == '__main__':
    main()
