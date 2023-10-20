import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Initialize the database to store models and their names
tetra_brick_database = defaultdict(str)

# Load the pre-trained object detection model
def load_object_detection_model():
    # You'll need to provide the path to your object detection model here
    model = tf.saved_model.load("path/to/object_detection/model")
    return model

# Load the pre-trained image classification model
def load_image_classification_model():
    # You'll need to provide the path to your image classification model here
    model = tf.keras.models.load_model("tetra_brick_basic_classifier.h5")
    return model

# Function to detect and classify Tetra-brick in an image
def detect_and_classify_tetra_brick(image_path):
    # Load the object detection model
    object_detection_model = load_object_detection_model()

    # Load the image classification model
    image_classification_model = load_image_classification_model()

    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)

    # Perform object detection
    detection_results = object_detection_model(image)

    # Assuming detection_results contains the bounding box and class label

    for result in detection_results['detection_boxes'][0]:
        x, y, x2, y2 = result
        # Crop the detected region
        roi = image[0, int(x * image.shape[1]):int(x2 * image.shape[1]), int(y * image.shape[2]):int(y2 * image.shape[2]), :]

        # Perform image classification on the cropped region
        classification_result = image_classification_model.predict(roi)

        # Check if the model already exists in the database
        model_name = tetra_brick_database.get(tuple(classification_result))
        if model_name:
            print("Detected as:", model_name)
        else:
            print("Tetra-brick not found in the database.")
            user_input = input("Please enter the name of the model: ")
            tetra_brick_database[tuple(classification_result)] = user_input
            print("Model added to the database as:", user_input)

# Function to compute the distance between two feature vectors
def compute_distance(vector1, vector2):
    return cosine_similarity([vector1], [vector2])[0][0]

# Function to calculate the minimum and maximum distances in the database
def calculate_min_max_distances():
    feature_vectors = list(tetra_brick_database.keys())
    min_distance = float('inf')
    max_distance = 0

    for i in range(len(feature_vectors)):
        for j in range(i + 1, len(feature_vectors)):
            dist = compute_distance(feature_vectors[i], feature_vectors[j])
            min_distance = min(min_distance, dist)
            max_distance = max(max_distance, dist)

    return min_distance, max_distance

# Function to calculate distances of a new model to existing models
def calculate_distances_to_existing_model(new_model_features):
    distances = {}
    for existing_features, model_name in tetra_brick_database.items():
        dist = compute_distance(new_model_features, existing_features)
        distances[model_name] = dist

    return distances

# Function to get the number of models in the database
def get_database_size():
    return len(tetra_brick_database)

# Example usage
image_path = "tcol1.bmp"
detect_and_classify_tetra_brick(image_path)
min_distance, max_distance = calculate_min_max_distances()
print("Number of models in the database:", get_database_size())
print("Minimum distance among models:", min_distance)
print("Maximum distance among models:", max_distance)
