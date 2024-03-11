#not working

import cv2 as cv
import math
import time
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
from PIL import Image

# # Import RetinaFace from the retinaface module
# from retinaface import RetinaFace  
from retinaface import RetinaFace as RF

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Initialize RetinaFace detector
detector = RF()

# Your existing code continues...
# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="dima806/facial_age_image_detection")

# Map classes to age ranges
class_to_age_range = {
    '01': '01',
    '02': '02',
    '03': '03',
    '04': '04',
    '05': '05',
    '06-07': '06-07',
    '08-09': '08-09',
    '10-12': '10-12',
    '13-15': '13-15',
    '16-20': '16-20',
    '21-25': '21-25',
    '26-30': '26-30',
    '31-35': '31-35',
    '36-40': '36-40',
    '41-45': '41-45',
    '46-50': '46-50',
    '51-55': '51-55',
    '56-60': '56-60',
    '61-65': '61-65',
    '66-70': '66-70',
    '71-80': '71-80',
    '81-90': '81-90',
    '90+': '90+'
}

def getFaceBox(frame):
    # Perform face detection using RetinaFace
    faces = detector.predict(frame)
    bboxes = []

    for face in faces:
        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
        bboxes.append([x1, y1, x2, y2])
        # Draw a green rectangle around the face
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return faces, bboxes


def age_gender_detector(frame):
    detected_faces, bboxes = getFaceBox(frame)
    
    for face in detected_faces:
        x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
        
        # Crop face from the frame
        face_img = frame[y1:y2, x1:x2]
        
        # Convert the cropped face to a PIL image
        img_pil = Image.fromarray(cv.cvtColor(face_img, cv.COLOR_BGR2RGB))
        
        # Perform age detection using the pipeline
        results = pipe(img_pil)
        
        # Extract the prediction
        prediction = results[0]
        
        # Get the predicted age label
        predicted_age_class = prediction['label']
        
        # Map predicted class to age range
        age_range = class_to_age_range.get(predicted_age_class, 'Unknown')
        
        # Display the prediction on the frame
        # Use relative coordinates to draw text on the frame
        cv.putText(frame, f"{age_range} (Conf: {prediction['score']:.2f})", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    
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
