from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the pixel values

# print(f"Max pixel value in training data: {x_train.max()}")
# print(f"Min pixel value in training data: {x_train.min()}")
# print(f"Max pixel value in test data: {x_test.max()}")
# print(f"Min pixel value in test data: {x_test.min()}")

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

y_pred = model.predict(x_test)
y_pred_classes = y_pred.argmax(axis=1)
# print(classification_report(y_test, y_pred_classes))

for batch_size in [32, 64, 128]:
    print(f"\nTraining with batch size: {batch_size}")
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, batch_size=batch_size, validation_split=0.2)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy with batch size {batch_size}: {test_acc:.4f}")
