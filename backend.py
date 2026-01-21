import os
from flask import Flask, request, render_template
from flask_cors import CORS
from test import predict_image  # Your prediction function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded.")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', prediction="Empty file name.")

    # Save uploaded image
    filename = 'uploaded_image.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(image_path)

    # Predict
    prediction_dict = predict_image(image_path)
    predicted_class = prediction_dict.get('predicted_class', prediction_dict)
    confidence = prediction_dict.get('confidence', None)

    prediction_text = f"{predicted_class}"
    if confidence is not None:
        prediction_text += f" ({confidence * 100:.2f}%)"

    return render_template('index.html', prediction=prediction_text, image_file='uploads/' + filename)

if __name__ == '__main__':
    app.run(debug=True)
