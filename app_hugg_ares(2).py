# this code classifies everyone in 16-20

import cv2 as cv
import math
import time
from flask import Flask, render_template, request
import torch
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="ares1123/photo_age_detection")

# Load pre-trained face detection model (VGGFace for example)
# You can replace this with any other face detection model
# Make sure to adjust the input size accordingly if needed
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

def getFaceBox(frame):
    # Convert frame to grayscale for face detection
    red = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(red, scaleFactor=1.3, minNeighbors=5)
    bboxes = []

    for (x, y, w, h) in faces:
        confidence = 1.0  # You can adjust confidence level based on the face detection method used
        bboxes.append([x, y, x + w, y + h, confidence])

    return faces, bboxes


def age_gender_detector(frame):
    detected_faces, bboxes = getFaceBox(frame)

    for (x, y, w, h), bbox in zip(detected_faces, bboxes):
        x1, y1, x2, y2, confidence = bbox
        
        # Crop face from the frame
        face = frame[y1:y2, x1:x2]
        
        # Convert the cropped face to a PIL image
        img_pil = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)
        
        # Perform age detection using the pipeline
        results = pipe(img_pil)
        
        # Extract the prediction
        prediction = results[0]
        
        # Get the predicted age label
        predicted_age = prediction['label']
        
        # Classify age into specified age ranges
        if predicted_age in ['26-30', '31-35']:
            age_group = '26-35'
        elif predicted_age in ['56-60', '41-45']:
            age_group = '41-60'
        else:
            age_group = '16-20'
        
        # Display the prediction on the frame
        # Use relative coordinates to draw text on the frame
        cv.putText(frame, f"{age_group} (Conf: {prediction['score']:.2f})", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box around detected face
    
    return frame


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader", methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file1']
        if f:
            f_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(f_path)
            input_image = cv.imread(f_path)
            if input_image is not None:
                output_image = age_gender_detector(input_image)
                if output_image is not None:
                    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
                    cv.imwrite(output_path, output_image)
                    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
                    pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
                    return render_template("uploaded.html", output_image=pic1, input_image=pic2)
                else:
                    return "Error processing image. Please try again."
            else:
                return "Error loading uploaded image. Please try again."
        else:
            return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=True)
