from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

print("Loading model...")

model = load_model('saved_models/skin_cancer_model1.h5')

print("Model loaded successfully.")

optimizer = Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')


# Check file extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Image preprocessing
def preprocess_image(image_path):

    img = image.load_img(image_path, target_size=(224, 224))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    return img


# Predict class
def classify_image(image_path):

    img = preprocess_image(image_path)

    prediction = model.predict(img)

    class_names = [
        'actinic keratosis',
        'basal cell carcinoma',
        'dermatofibroma',
        'melanoma',
        'nevus',
        'pigmented benign keratosis',
        'seborrheic keratosis',
        'squamous cell carcinoma',
        'vascular lesion'
    ]

    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class


# Benign or Malignant
def determine_result_class(predicted_class):

    benign_classes = [
        'actinic keratosis',
        'nevus',
        'pigmented benign keratosis',
        'seborrheic keratosis'
    ]

    if predicted_class in benign_classes:
        return 'Benign'
    else:
        return 'Malignant'


# Upload Route
@app.route('/upload', methods=['POST'])
def upload_file():

    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        predicted_class = classify_image(filepath)

        result_class = determine_result_class(predicted_class)

        image_url = url_for('uploaded_file', filename=file.filename)

        return render_template(
            'result.html',
            predicted_class=predicted_class,
            result_class=result_class,
            image=image_url
        )

    return redirect(request.url)


# Display uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
