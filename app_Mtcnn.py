#install mtcnn using py -m pip install mtcnn opencv-python-headless
#install flask using py -m pip install flask
#install werkzeug using py -m pip install werkzeug
#isntall tensorflow using py -m pip install tensorflow
#upgrade tensorflow using py -m pip install --upgrade tensorflow
#upgrade pip using py -m pip install --upgrade pip

from mtcnn import MTCNN
import cv2 as cv
import math
import time
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os


def getFaceBox(detector, frame):
    if frame is None:
        return None, []

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Use MTCNN for face detection
    faces = detector.detect_faces(frameOpencvDnn)
    bboxes = []

    for face in faces:
        x, y, width, height = face['box']
        confidence = face['confidence']
        bboxes.append([x, y, x + width, y + height, confidence])
        cv.rectangle(frameOpencvDnn, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
    return frameOpencvDnn, bboxes


# faceProto = "models/opencv_face_detector.pbtxt"
# faceModel ="models/opencv_face_detector_uint8.pb"
detector = MTCNN()

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

#ageList = ['(0-2)', '(3-5)', '(6-9)', '(10-13)', '(14-17)', '(18-21)', '(22-25)', '(26-30)', '(31-35)', '(36-40)', '(41-45)', '(46-50)', '(51-55)', '(56-60)', '(61-)']
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-)']
genderList = ['Male', 'Female']

ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
# faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20

def age_gender_detector(frame):
    t = time.time()
    frameFace, bboxes = getFaceBox(detector, frame)
    for bbox in bboxes:
        face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        confidence = bbox[4]  # Confidence value from bounding box
        label = "{}, {} (Confidence: {:.2f})".format(gender, age, confidence)
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
    return frameFace

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/uploader", methods=['POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file1']
        if f:
            # Rename the uploaded file to 'image.jpg'
            f.filename = 'image.jpg'
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
