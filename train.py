import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
import joblib


def save_features_to_excel(features, labels, class_names, output_file):
    """
    Save all features for each image along with the class name to an Excel file.

    Args:
        features (numpy.ndarray): Array of features extracted for all images.
        labels (numpy.ndarray): Array of class labels corresponding to the features.
        class_names (list): List of class names corresponding to the labels.
        output_file (str): File path to save the Excel file.
    """
    # Define column names for the features
    feature_columns = [
        "Texture_Contrast", "Texture_Dissimilarity", "Texture_Homogeneity",
        "Texture_Energy", "Texture_Correlation", "Color_Histogram",
        "Shape_Area", "Shape_Perimeter", "Shape_Circularity"
    ]
    
    # Prepare data for the DataFrame
    data = []
    for feature, label in zip(features, labels):
        row = {**dict(zip(feature_columns, feature)), "Class_Name": class_names[label]}
        data.append(row)

    # Convert to a pandas DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Features for all images saved to {output_file}")


# Function to extract texture features using GLCM
def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

# Function to extract color histogram features (single averaged value)
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return np.mean(hist)

# Function to extract shape features
def extract_shape_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(3)  # Return zero features if no contours found

    contour = max(contours, key=cv2.contourArea)

    # Visualize contours for debugging
    # visualize_contours(image, contour)

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Normalize by image size
    height, width = gray_image.shape
    normalized_area = area / (height * width)
    normalized_perimeter = perimeter / (2 * (height + width))
    circularity = 4 * np.pi * (normalized_area / (normalized_perimeter ** 2)) if normalized_perimeter > 0 else 0

    return np.array([normalized_area, normalized_perimeter, circularity])

# Preprocessing function
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be opened: {image_path}")

        # Extract shape features from the original image
        shape_features = extract_shape_features(image)
        
        # Resize image for CNN input
        image_resized = cv2.resize(image, (224, 224))

        # Extract other features from the resized image
        texture_features = extract_texture_features(image_resized)
        color_feature = extract_color_histogram(image_resized)

        # Combine features
        traditional_features = np.concatenate([texture_features, [color_feature], shape_features])

        return image_resized, traditional_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


# Function to summarize features by class
def summarize_features_by_class(features, labels, class_names, output_file):
    summary_data = []
    for class_idx, class_name in enumerate(class_names):
        class_features = features[labels == class_idx]
        if len(class_features) == 0:
            continue
        
        feature_means = np.mean(class_features, axis=0)
        feature_mins = np.min(class_features, axis=0)
        feature_maxs = np.max(class_features, axis=0)
        
        summary_row = {
            "Class": class_name,
            "Feature Means": list(feature_means),
            "Feature Min": list(feature_mins),
            "Feature Max": list(feature_maxs),
        }
        summary_data.append(summary_row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_excel(output_file, index=False)
    print(f"Feature summary saved to {output_file}")

# Function to create the improved model
def create_combined_model(cnn_input_shape, num_traditional_features, num_classes):
    # CNN branch
    cnn_input = Input(shape=cnn_input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    cnn_output = Dense(64, activation='relu')(x)

    # Traditional features branch
    traditional_input = Input(shape=(num_traditional_features,))
    y = Dense(64, activation='relu')(traditional_input)

    # Combine branches
    combined = tf.keras.layers.concatenate([cnn_output, y])
    combined = Dense(128, activation='relu')(combined)
    final_output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[cnn_input, traditional_input], outputs=final_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Directory where dataset is located
dataset_dir = r'D:\Raw_Potato_2020 - Copy'
class_names = os.listdir(dataset_dir)

# Prepare dataset
images = []
traditional_features = []
labels = []

for label in class_names:
    class_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img, features = preprocess_image(img_path)
            if img is not None and features is not None:
                images.append(img)
                traditional_features.append(features)
                labels.append(class_names.index(label))

# Convert to numpy arrays
images = np.array(images)
traditional_features = np.array(traditional_features)
labels = np.array(labels)

if images.size == 0:
    print("No images were loaded. Please check your dataset directory and files.")
    exit()

# Normalize images and standardize traditional features
images = images.astype('float32') / 255.0
scaler = StandardScaler()
traditional_features = scaler.fit_transform(traditional_features)

# Save scaler and class names
joblib.dump(scaler, 'scaler.pkl')
with open('class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

# Split the data
train_images, val_images, train_features, val_features, train_labels, val_labels = train_test_split(
    images, traditional_features, labels, test_size=0.2, random_state=42
)

# Define model
cnn_input_shape = (224, 224, 3)
num_classes = len(class_names)
num_traditional_features = traditional_features.shape[1]
model = create_combined_model(cnn_input_shape, num_traditional_features, num_classes)

# Train model
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices(((train_images, train_features), train_labels)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(((val_images, val_features), val_labels)).batch(batch_size)

model.fit(train_dataset, epochs=20, validation_data=val_dataset)

# Save model
model.save("model.keras")
print("Model saved successfully in Keras format.")

# Define the output file path
output_file = "image_features_with_classes.xlsx"

# Call the function to save the features
save_features_to_excel(traditional_features, labels, class_names, output_file)
