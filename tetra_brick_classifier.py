import cv2
import numpy as np
import os
import sys
import json

# Define the threshold for considering a model as new
threshold = 0.001  # You can adjust this value based on your needs

# Function to extract SIFT features from an image
def extract_features(image):
    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute SIFT keypoints and descriptors
    kp, des = sift.detectAndCompute(image, None)

    return des

# Function to add a new tetra-brick model to the database
def add_to_database(database, name, feature_vector):
    database[name] = feature_vector if feature_vector is not None else None

# Function to classify a tetra-brick image using SIFT features
def classify_tetra_brick(image, database):
    if not database:
        return "No tetra-brick models in the database. Please add models."

    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to enhance edge features
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply the Hough Transform to detect lines (tetra-bricks)
    min_votes = 100  # Adjust the minimum number of votes as needed
    lines = cv2.HoughLines(edges, 1, np.pi / 90, min_votes)

    if lines is not None:
        # Once you detect a tetra-brick, extract SIFT features and use SIFT for classification
        features = extract_features(image)

        best_match = None
        best_match_distance = float('inf')

        for name, feature in database.items():
            if feature is not None:
                # Ensure feature vectors have the same type (uint8) and dimensions
                features = features.astype(np.uint8)
                feature = feature.astype(np.uint8)
                
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(features, feature, k=2)
                good_matches = []

                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

                distance = len(good_matches)

                if distance < best_match_distance:
                    best_match = name
                    best_match_distance = distance

        # If the best match's distance is above the defined threshold, it's considered a new model
        if best_match_distance > threshold:
            return "New tetra-brick model. Please enter its name."
        else:
            return best_match
    else:
        return "No tetra-brick models detected"

# Function to enter debug mode and display database statistics
def debug_mode(database, new_image=None):
    print("Number of tetra-brick models in the database:", len(database))

    if len(database) > 1:
        min_distance = float('inf')
        max_distance = 0

        # Calculate the minimum and maximum distances among existing models
        for model1 in database:
            for model2 in database:
                if model1 != model2 and database[model1] is not None and database[model2] is not None:
                    feature1 = database[model1]
                    feature2 = database[model2]

                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(feature1, feature2, k=2)
                    good_matches = []

                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                    distance = len(good_matches)

                    min_distance = min(min_distance, distance)
                    max_distance = max(max_distance, distance)

        print("Minimum distance among models:", min_distance)
        print("Maximum distance among models:", max_distance)

    # If a new image is provided, calculate its distance to existing models
    if new_image is not None:
        features = extract_features(new_image)
        distances_to_existing = {name: classify_tetra_brick(new_image, {name: feature}) if feature is not None else "No data" for name, feature in database.items()}
        print("Distances of the new image to existing models:")
        for model, distance in distances_to_existing.items():
            print(f"{model}: {distance}")

# Function to save the database to a file
def save_database(database, filename):
    with open(filename, 'w') as file:
        # Convert the feature vectors to lists before saving
        serializable_database = {name: feature.tolist() if feature is not None else None for name, feature in database.items()}
        json.dump(serializable_database, file)

# Function to load the database from a file or create an empty one if it doesn't exist
def load_database(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            # Load the serialized database and convert the feature lists back to NumPy arrays
            serialized_database = json.load(file)
            database = {name: np.array(feature) if feature else None for name, feature in serialized_database.items()}
            return database
    else:
        return {}

if __name__ == "__main__":
    # Specify the filename for the database file
    database_filename = 'tetra_brick_database.json'

    # Load the existing database from the file or create an empty one
    database = load_database(database_filename)

    if len(sys.argv) > 1:
        if sys.argv[1] == "d":
            debug_mode(database)
        else:
            input_file = sys.argv[1]
            image = cv2.imread(input_file)

            if image is not None:
                model_name = classify_tetra_brick(image, database)

                if model_name == "No tetra-brick models in the database. Please add models." or model_name == "New tetra-brick model. Please enter its name.":
                    print("New tetra-brick model detected. Please enter its name:")
                    new_model_name = input()
                    add_to_database(database, new_model_name, extract_features(image))
                    save_database(database, database_filename)  # Save the updated database
                    print("New model added to the database.")
                else:
                    print("Detected tetra-brick model:", model_name)

                if len(sys.argv) > 2 and sys.argv[2] == "d":
                    debug_mode(database, image)  # Enter debug mode with the new image
            else:
                print("Error: Unable to load the input image.")
