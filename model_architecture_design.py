import numpy as np
import os
import cv2
import tensorflow as tf
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

def preprocess_and_load_images(directory_path, image_size=(224, 224)):
    """
    Load and preprocess images from a directory.

    Args:
        directory_path (str): Path to the directory containing the images.
        image_size (tuple): Target size to resize images (default: (224, 224)).

    Returns:
        numpy.ndarray: Array of preprocessed images.
    """

    # Initialize an empty list to store preprocessed images
    preprocessed_images = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):
            # Load the image using OpenCV
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)

            # Resize the image to the specified size (e.g., 224x224)
            image = cv2.resize(image, image_size)

            # Normalize pixel values to the range [0, 1]
            image = image.astype("float32") / 255.0

            # Append the preprocessed image to the list
            preprocessed_images.append(image)


    # Convert the list of images to a NumPy array
    return np.array(preprocessed_images)


# Define the number of classes for your classification task
num_classes = 7  # Replace with the actual number of classes

# Create the MobileNet model with specified input shape
base_model = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)

# Add custom layers for your specific classification task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

# Output layer with the number of classes
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Load and preprocess your training images into train_data
# Normalize, resize, and rescale images to the required input size
train_data = preprocess_and_load_images("C:\\Users\\cferr\\OneDrive\\Escritorio\\tetra\\training_set")

# After loading the training data, add the following print statement
print("Number of training samples:", len(train_data))

# Print the shape of a sample image (assuming it's a NumPy array)
if len(train_data) > 0:
    print("Shape of a sample image:", train_data[0].shape)

# Create an array of labels corresponding to each image
# Use one-hot encoding for the labels
train_labels = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # Label for the first class
    [0, 1, 0, 0, 0, 0, 0],  # Label for the second class
    [0, 0, 1, 0, 0, 0, 0],  # Label for the third class
    [0 ,0, 0, 1, 0, 0, 0],  # Label for the fourth class
    [0 ,0, 0, 0, 1, 0, 0],  # Label for the fifth class
    [0 ,0, 0, 0, 0, 1, 0],  # Label for the sixth class
    [0 ,0, 0, 0, 0, 0, 1]   # Label for the seventh class
])

# Load and preprocess your validation images and labels similarly
validation_data = preprocess_and_load_images("C:\\Users\\cferr\\OneDrive\\Escritorio\\tetra\\validation_set")

# After loading the training data, add the following print statement
print("Number of training samples:", len(validation_data))

# Print the shape of a sample image (assuming it's a NumPy array)
if len(validation_data) > 0:
    print("Shape of a sample image:", validation_data[0].shape)


# Create an array of labels corresponding to each image
# Use one-hot encoding for the labels
validation_labels = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # Label for the first class
    [0, 1, 0, 0, 0, 0, 0],  # Label for the second class
    [0, 0, 1, 0, 0, 0, 0],  # Label for the third class
    [0 ,0, 0, 1, 0, 0, 0],  # Label for the fourth class
    [0 ,0, 0, 0, 1, 0, 0],  # Label for the fifth class
    [0 ,0, 0, 0, 0, 1, 0],  # Label for the sixth class
    [0 ,0, 0, 0, 0, 0, 1]   # Label for the seventh class
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, validation_data=(validation_data, validation_labels), epochs=20, batch_size=32)

# Save the trained model to a file
model.save("tetra_brick_basic_classifier.h5")