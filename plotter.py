import pickle
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Function to load data
def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# List of filenames
file_names = ["all_rounds_info20clients25framewidth1epoch.pkl",
              "all_rounds_info20clients20framewidth1epoch.pkl",
"memory_info20clients15framewidth1epoch.pkl",
"all_rounds_info20clients10framewidth1epoch.pkl",
"all_rounds_info20clients5framewidth1epoch.pkl",
"all_rounds_info20clients2framewidth1epoch.pkl"]

# Plotting setup
plt.figure(figsize=(12, 6))

# Loop through each file and plot its data
for file_name in file_names:
    all_rounds_info = load_data(file_name)

    # Extract accuracy from the results
    rounds = all_rounds_info['rounds']
    accuracies = all_rounds_info['global_accuracies']

    # Extract frame width from file name for the legend
    frame_width = file_name.split('clients')[1].split('epoch')[0]

    # Plot accuracy
    plt.plot(rounds, accuracies, marker='o', label=f'Frame Width {frame_width}')

# Finalize plot
plt.xlabel('Rounds')
plt.ylabel('Accuracy')
plt.title('Accuracy over Rounds for Different Frame Widths')
plt.grid(True)
plt.legend()

# Save plot to a PNG file
plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
