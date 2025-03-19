import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
layers = tf.keras.layers  # Access layers from tf.keras

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

img_height = 128
img_width = 128
batch_size = 32

# Load the datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Store class names before caching and prefetching
class_names = train_ds.class_names

# Cache and prefetch for performance optimization
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define the model
model = tf.keras.Sequential([
    layers.Rescaling(1./255),  # Rescaling layer
    layers.Conv2D(32, 3, activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Output layer for binary classification
])

# Function to visualize predictions
def display_predictions(dataset, model, class_names, num_images=32):
    plt.figure(figsize=(32, 32))

    # Loop over one batch of images and labels from the dataset
    for images, labels in dataset.take(1):  # Take one batch
        predictions = model.predict(images)  # Get predictions for this batch
        predicted_labels = np.argmax(predictions, axis=-1)  # Get the predicted class label

        for i in range(min(num_images, len(images))):  # Loop over the images in the batch
            plt.subplot(11, 3, i + 1)
            plt.imshow(np.squeeze(images[i].numpy().astype("uint8")), cmap="gray")  # Show image
            plt.title(f"True: {class_names[labels[i]]}, Pred: {class_names[predicted_labels[i]]}")
            plt.axis("off")

    plt.show()

# Call the function to display predictions
display_predictions(test_ds, model, class_names, num_images=32)  # Adjust `num_images` to show more/less

