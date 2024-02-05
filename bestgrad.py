# Set up logging configuration
import logging
logging.basicConfig(filename='simplifiedFL.log', level=logging.INFO, format='%(message)s')
import tensorflow as tf

# Check if TensorFlow is able to detect the GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("No GPU detected. Running on CPU.")

import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(5)
tf.random.set_seed(5)

def train_validation_split(X_train, Y_train):
    train_length = len(X_train)
    validation_length = int(train_length / 4)
    X_validation = X_train[:validation_length]
    X_train = X_train[validation_length:]
    Y_validation = Y_train[:validation_length]
    Y_train = Y_train[validation_length:]
    return X_train, Y_train, X_validation, Y_validation

# Network settings
learning_rate = 0.01
epochs = 1
batch = 32
iterations = 25
number_of_users = 20

# Load and preprocess CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
Y_train, Y_test = to_categorical(Y_train), to_categorical(Y_test)

# FL setting
size_of_user_ds = int(len(X_train)/number_of_users)
train_data_X = np.zeros((number_of_users,size_of_user_ds, 32, 32, 3))
train_data_Y = np.ones((number_of_users,size_of_user_ds,10))

for i in range(number_of_users):
    train_data_X[i] = X_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]
    train_data_Y[i] = Y_train[size_of_user_ds*i:size_of_user_ds*i+size_of_user_ds]



# Model definition
initializer = tf.keras.initializers.HeUniform(seed=5)
model = models.Sequential([
    # Add layers as per the provided structure
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same', input_shape=(32, 32, 3)),
  layers.BatchNormalization(),
  layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.2),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.3),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5), padding='same'),
  layers.BatchNormalization(),
  layers.MaxPooling2D((2, 2)),
  layers.Dropout(0.4),
  layers.Flatten(),
  layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=5)),
  layers.BatchNormalization(),
  layers.Dropout(0.5),
  layers.Dense(10, activation='softmax'),
])

model_name = model._name
opt = Adam(learning_rate=0.0001) if model_name not in ['VGG16', 'resnet18'] else Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('federated_modelavg2users.h5')

# Federated Learning Process
wc = model.get_weights()  # Initial global model weights
logging.info(f"\n Simulating with number_of_users: {number_of_users}")
best_val_accuracy = 0  # Initialize the best validation accuracy
framesize = 2
for iter in range(iterations):
    gradients = []
    total_data_points = 0
    best_gradient = None  # Initialize the best gradient
    winit = model.get_weights()
    for slot in range(framesize):
        # Each user makes a decision to transmit or not in this slot
        decisions = np.random.choice([0, 1], size=number_of_users, p=[1-1/number_of_users, 1/number_of_users])
        # Check if exactly one user decided to transmit
        if np.sum(decisions) == 1:
            i = np.where(decisions == 1)[0][0]  # Get the index of transmitting user
            X_train_u = train_data_X[i]
            Y_train_u = train_data_Y[i]
            # Train model for each user
            model.set_weights(wc)
            model.fit(X_train_u, Y_train_u, epochs=epochs, batch_size=batch, shuffle=False)
            # After training, evaluate on the validation set
            val_loss, val_accuracy = model.evaluate(X_test, Y_test, verbose=0)
            logging.info(f"User {i}, Validation Accuracy: {val_accuracy:.4f}")
            # Check if this model is the best so far based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy  # Update the best validation accuracy
                # Calculate gradient for the best model
                wu = model.get_weights()
                best_gradient = [wu[j] - wc[j] for j in range(len(wc))]  # Store the best gradient
    # Apply the best gradient if one was selected
    if best_gradient is not None:
        wc = [wc[i] + best_gradient[i] for i in range(len(wc))]
        model.set_weights(wc)  # Update the global model with the best gradient
        # Evaluate global model
        _, accuracy = model.evaluate(X_test, Y_test)
        logging.info(f"Iteration {iter + 1}, Accuracy: {accuracy:.4f}")
    else:
        model.set_weights(winit)



# Save final model
model.save("federated_model2users.h5")

