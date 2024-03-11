#install mtcnn using py -m pip install mtcnn opencv-python-headless
#install flask using py -m pip install flask
#install werkzeug using py -m pip install werkzeug
#isntall tensorflow using py -m pip install tensorflow
#upgrade tensorflow using py -m pip install --upgrade tensorflow
#upgrade pip using py -m pip install --upgrade pip

import cv2 as cv
import math
import time
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os


def getFaceBox(net, frame, conf_threshold=0.7):
    if frame is None:
        return None, []

    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True)
    net.setInput(blob)
    detections = net.forward()
    bboxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2, confidence])  # Include confidence
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


faceProto = "models/opencv_face_detector.pbtxt"
faceModel ="models/opencv_face_detector_uint8.pb"

ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"

genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-)']

genderList = ['Male', 'Female']

ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20

def age_gender_detector(frame):
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
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
# def uploader():
#     if request.method == 'POST':
#         f = request.files['file1']
#         if f:
#             f_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
#             f.save(f_path)
#             input_image = cv.imread(f_path)
#             if input_image is not None:
#                 output_image = age_gender_detector(input_image)
#                 if output_image is not None:
#                     output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
#                     cv.imwrite(output_path, output_image)
#                     pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
#                     pic2 = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
#                     return render_template("uploaded.html", output_image=pic1, input_image=pic2)
#                 else:
#                     return "Error processing image. Please try again."
#             else:
#                 return "Error loading uploaded image. Please try again."
#         else:
#             return "No file uploaded."
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
