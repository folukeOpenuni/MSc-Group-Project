from mtcnn import MTCNN
import cv2 as cv
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Initialize MTCNN detector
detector = MTCNN()

# Load the VGGFace model for face recognition
vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load the age detection model
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
ageNet = cv.dnn.readNet(ageModel, ageProto)

def detect_age(frame):
    frame_opencv_dnn = frame.copy()

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame_opencv_dnn)

    for face in faces:
        x, y, width, height = face['box']
        face_region = frame_opencv_dnn[y:y+height, x:x+width]

        # Preprocess face region for VGGFace
        face_region_resized = cv.resize(face_region, (224, 224))
        face_region_preprocessed = preprocess_input(face_region_resized)

        # Extract features using VGGFace
        features = vggface_model.predict(face_region_preprocessed.reshape(1, 224, 224, 3))

        # Predict age using the age detection model
        blob = cv.dnn.blobFromImage(face_region_preprocessed, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = agePreds[0].argmax()

        # Draw age prediction on the frame
        cv.putText(frame_opencv_dnn, f"Age: {age}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame_opencv_dnn


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
            f_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(f_path)
            input_image = cv.imread(f_path)
            if input_image is not None:
                output_image = detect_age(input_image)
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
