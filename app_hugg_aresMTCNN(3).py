#ares mtcnn classifier clasifies everyones age into categories 16-20

import cv2 as cv
import math
import time
from flask import Flask, render_template, request
import torch
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
from PIL import Image
from mtcnn import MTCNN  # Import MTCNN for face detection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="ares1123/photo_age_detection")

# Initialize MTCNN for face detection
detector = MTCNN()

def getFaceBox(frame):
    # Perform face detection using MTCNN
    detected_faces = detector.detect_faces(frame)
    bboxes = []

    print(f"Detected faces: {detected_faces}")

    for face in detected_faces:
        if 'box' in face:
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            bboxes.append([x1, y1, x2, y2])

    print(f"Bboxes: {bboxes}")

    return detected_faces, bboxes


def age_gender_detector(frame):
    detected_faces, bboxes = getFaceBox(frame)
    
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        
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
    
    return frame


@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader", methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file1']
        if f:
            # Save the file with its original filename
            f_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(f_path)
            input_image = cv.imread(f_path)
            if input_image is not None:
                output_image = age_gender_detector(input_image)
                if output_image is not None:
                    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
                    cv.imwrite(output_path, output_image)
                    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
                    # Use the original filename for the input image
                    pic2 = f_path
                    return render_template("uploaded.html", output_image=pic1, input_image=pic2)
                else:
                    return "Error processing image. Please try again."
            else:
                return "Error loading uploaded image. Please try again."
        else:
            return "No file uploaded."

if __name__ == '__main__':
    app.run(debug=True)