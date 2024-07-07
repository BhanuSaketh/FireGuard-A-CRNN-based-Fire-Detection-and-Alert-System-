from flask import Flask, render_template, request, Response, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__, static_url_path='/static')

# Load the CRNN model
model = load_model('crnn_model_30.h5')
files = ['fire.mp4', 'green.mp4', 'fog.mp4']

# Action labels
action_labels = {
    0: "burned-area",
    1: "fire-smoke",
    2: "fog-area",
    3: "green-area"
}

# Function to preprocess frames
def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype("float") / 255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    return frame

# Function to predict action label for a frame
def predict_action(frame):
    frame = preprocess_frame(frame, (224, 224))
    preds = model.predict(frame)
    action_label_index = np.argmax(preds, axis=1)[0]
    action_label = action_labels[action_label_index]
    return action_label

# Function to generate frames
def generate_frames(video_choice):
    if video_choice == '4':
        cap = cv2.VideoCapture(0)  # Capture live video from webcam
    else:
        video_path = files[int(video_choice) - 1]
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        yield "Error: Couldn't open video file."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        action_label = predict_action(frame)

        if action_label == "fire-smoke":
            # Trigger the alert
            yield "fire-smoke"

        cv2.putText(frame, "Predicted Action: {}".format(action_label), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('web.html')

@app.route('/process', methods=['POST'])
def process():
    video_choice = request.form['video_choice']
    return redirect(url_for('video_feed', video_choice=video_choice))

@app.route('/video_feed')
def video_feed():
    video_choice = request.args.get('video_choice')
    if video_choice is None:
        return "Error: Video choice is missing."
    return Response(generate_frames(video_choice), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
