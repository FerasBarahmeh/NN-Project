from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

# Load and preprocess the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the pixel values


# verify normalization
print(f"Max pixel value in training data: {x_train.max()}")
print(f"Min pixel value in training data: {x_train.min()}")
print(f"Max pixel value in test data: {x_test.max()}")
print(f"Min pixel value in test data: {x_test.min()}")


# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten the input images to a 1D array
    layers.Dense(128, activation='relu'),  # First hidden layer with ReLU activation
    layers.Dropout(0.5),  # Dropout regularization to prevent overfitting
    layers.Dense(10, activation='softmax')  # Output layer with softmax activation for classification
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer
              loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification
              metrics=['accuracy'])  # Metrics to monitor during training

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)  # Train the model with 20% validation split

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Generate classification report
y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred_classes))

# Basic Hyperparameter Tuning Example: Different Batch Sizes
for batch_size in [32, 64, 128]:
    print(f"\nTraining with batch size: {batch_size}")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy with batch size {batch_size}: {test_acc:.4f}")
