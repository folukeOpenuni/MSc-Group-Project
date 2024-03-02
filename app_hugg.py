# #import torch using the following command pip install torch

# import cv2 as cv
# import math
# import time
# from flask import Flask, render_template, request
# import torch
# from werkzeug.utils import secure_filename
# import os
# from transformers import AutoImageProcessor, AutoModelForImageClassification


# def getFaceBox(net, frame, conf_threshold=0.7):
#     if frame is None:
#         return None, []

#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True)
#     net.setInput(blob)
#     detections = net.forward()
#     bboxes = []

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             bboxes.append([x1, y1, x2, y2, confidence])  # Include confidence
#             cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn, bboxes


# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-)']
# genderList = ['Male', 'Female']

# faceProto = "models/opencv_face_detector.pbtxt"
# faceModel ="models/opencv_face_detector_uint8.pb"

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# age_processor = AutoImageProcessor.from_pretrained("dima806/faces_age_detection")
# age_model = AutoModelForImageClassification.from_pretrained("dima806/faces_age_detection")

# faceNet = cv.dnn.readNet(faceModel, faceProto)

# padding = 20

# def age_gender_detector(frame):
#     t = time.time()
#     frameFace, bboxes = getFaceBox(faceNet, frame)
#     for bbox in bboxes:
#         face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
#         age_input = age_processor(face, return_tensors="pt")
#         age_output = age_model(**age_input)
#         predicted_age_index = torch.argmax(age_output.logits)
#         predicted_age = ageList[predicted_age_index]
#         confidence = bbox[4]  # Confidence value from bounding box
#         gender = "Not implemented"  # Gender detection is not implemented in this code
#         label = "{}, {} (Confidence: {:.2f})".format(gender, predicted_age, confidence)
#         cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
#     return frameFace

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route("/uploader", methods=['POST'])
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

# if __name__ == '__main__':
#     app.run(debug=True)

# import cv2 as cv
# import math
# import time
# from flask import Flask, render_template, request
# import torch
# from werkzeug.utils import secure_filename
# import os
# from PIL import Image
# from transformers import pipeline

# # Initialize the pipeline
# pipe = pipeline("image-classification", model="dima806/faces_age_detection")

# faceProto = "models/opencv_face_detector.pbtxt"
# faceModel ="models/opencv_face_detector_uint8.pb"

# MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# faceNet = cv.dnn.readNet(faceModel, faceProto)

# padding = 20

# def getFaceBox(net, frame, conf_threshold=0.7):
#     if frame is None:
#         return None, []

#     frameOpencvDnn = frame.copy()
#     frameHeight = frameOpencvDnn.shape[0]
#     frameWidth = frameOpencvDnn.shape[1]
#     blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True)
#     net.setInput(blob)
#     detections = net.forward()
#     bboxes = []

#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > conf_threshold:
#             x1 = int(detections[0, 0, i, 3] * frameWidth)
#             y1 = int(detections[0, 0, i, 4] * frameHeight)
#             x2 = int(detections[0, 0, i, 5] * frameWidth)
#             y2 = int(detections[0, 0, i, 6] * frameHeight)
#             bboxes.append([x1, y1, x2, y2, confidence])  # Include confidence
#             cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn, bboxes

# def age_gender_detector(frame):
#     t = time.time()
#     frameFace, bboxes = getFaceBox(faceNet, frame)
#     for bbox in bboxes:
#         face = frame[max(0, bbox[1]-padding):min(bbox[3]+padding, frame.shape[0]-1), max(0, bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
#         # Convert numpy array to PIL image
#         pil_image = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
#         age_prediction = pipe(pil_image)[0]
#         predicted_age = age_prediction['score']
#         confidence = bbox[4]  # Confidence value from bounding box
#         gender = "Not implemented"  # Gender detection is not implemented in this code
#         label = "{}, {} years old (Confidence: {:.2f})".format(gender, predicted_age, confidence)
#         cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
#     return frameFace

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route("/uploader", methods=['POST'])
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

# if __name__ == '__main__':
#     app.run(debug=True)
import cv2 as cv
import math
import time
from flask import Flask, render_template, request
import torch
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
from PIL import Image  # Import the Image module from PIL

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Use a pipeline as a high-level helper
pipe = pipeline("image-classification", model="dima806/faces_age_detection")

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
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

    return frameOpencvDnn, bboxes

def age_gender_detector(frame):
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    for bbox in bboxes:
        x1, y1, x2, y2, confidence = bbox
        
        # Crop face from the frame
        face = frame[y1:y2, x1:x2]
        
        # Convert the cropped face to a PIL image
        img_pil = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)
        
        # Perform age detection using the pipeline
        results = pipe(img_pil)
        
        # Print the predicted labels and their scores
        print(results)
        
        # Extract the prediction
        prediction = results[0]
        
        # Process the prediction
        gender = "Not implemented"  # Gender detection is not implemented in this code
        predicted_age = prediction['label']
        confidence = prediction['score']
        
        # Display the prediction on the frame
        cv.putText(frameFace, f"{predicted_age} (Conf: {confidence:.2f})", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    
    return frameFace

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
    # Initialize the face detection model
    faceProto = "models/opencv_face_detector.pbtxt"
    faceModel ="models/opencv_face_detector_uint8.pb"
    faceNet = cv.dnn.readNet(faceModel, faceProto)
    
    app.run(debug=True)


