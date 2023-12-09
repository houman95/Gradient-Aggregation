import pickle
import matplotlib.pyplot as plt
import torch
# Load the results
# Function to load data with mapping to CPU if CUDA is not available
# Function to load data
def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Load the results with appropriate device mapping
all_rounds_info = load_data('all_rounds_info.pkl')
# Load the results with appropriate device mapping
all_rounds_info = load_data('all_rounds_info.pkl')

# Extract accuracy and loss from the results
rounds = all_rounds_info['rounds']
accuracies = all_rounds_info['global_accuracies']
losses = all_rounds_info['global_losses']

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(rounds, accuracies, marker='o', color='b', label='Accuracy')
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy over Rounds')
plt.grid(True)
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(rounds, losses, marker='x', color='r', label='Loss')
plt.xlabel('Rounds')
plt.ylabel('Loss')
plt.title('Loss over Rounds')
plt.grid(True)
plt.legend()

# Save plot to a PNG file
plt.tight_layout()
plt.savefig('federated_learning_results.png', dpi=300)
