import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from tkinter import Tk, filedialog

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Function to open file dialog and get image path
def select_image():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(title="Select a Potato Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path

# Function to extract texture features
def extract_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features = {
        "Contrast": graycoprops(glcm, 'contrast')[0, 0],
        "Dissimilarity": graycoprops(glcm, 'dissimilarity')[0, 0],
        "Homogeneity": graycoprops(glcm, 'homogeneity')[0, 0],
        "Energy": graycoprops(glcm, 'energy')[0, 0],
        "Correlation": graycoprops(glcm, 'correlation')[0, 0],
    }
    return features

# Function to extract color histogram features
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return {"Average Color Histogram Value": np.mean(hist)}

# Function to extract shape features
def extract_shape_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {"Area": 0, "Perimeter": 0, "Circularity": 0}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    height, width = gray_image.shape
    normalized_area = area / (height * width)
    normalized_perimeter = perimeter / (2 * (height + width))
    circularity = 4 * np.pi * (normalized_area / (normalized_perimeter ** 2)) if normalized_perimeter > 0 else 0

    return {
        "Area": normalized_area,
        "Perimeter": normalized_perimeter,
        "Circularity": circularity,
    }

# Image preprocessing function
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found or cannot be opened: {image_path}")
        
        image_resized = cv2.resize(image, (224, 224))
        texture_features = extract_texture_features(image_resized)
        color_features = extract_color_histogram(image_resized)
        shape_features = extract_shape_features(image)
        
        all_features = {**texture_features, **color_features, **shape_features}
        traditional_features = np.array(list(all_features.values()))

        return image_resized, traditional_features, all_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None

# Load the trained model, scaler, and class names
model = load_model("model.keras")
scaler = joblib.load("scaler.pkl")
with open("class_names.txt", 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Prediction function
def predict_image(image_path):
    try:
        image_resized, traditional_features, all_features = preprocess_image(image_path)
        if image_resized is None or traditional_features is None:
            return {"error": "Error in preprocessing. Unable to predict."}
        
        image_resized = image_resized.astype('float32') / 255.0
        traditional_features = scaler.transform([traditional_features])
        
        prediction = model.predict([np.expand_dims(image_resized, axis=0), traditional_features])
        predicted_class_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_class_idx]
        confidence = prediction[0][predicted_class_idx]
        
        result = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "extracted_features": all_features,
        }
        return result
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Run the program
if __name__ == "__main__":
    image_path = select_image()
    if image_path:
        result = predict_image(image_path)
        print(result)
    else:
        print("No image selected.")
