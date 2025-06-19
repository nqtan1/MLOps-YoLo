from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import os
import shutil
import cv2 
import psutil  # Get system information
from prometheus_client import Gauge, generate_latest
from pathlib import Path

app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO('yolov8n.pt')  

# Initialize Prometheus metrics
cpu_usage = Gauge('cpu_usage', 'CPU Usage')
memory_usage = Gauge('memory_usage', 'Memory Usage')
disk_usage = Gauge('disk_usage', 'Disk Usage')

@app.route('/metrics')
def metrics():
    cpu_usage.set(psutil.cpu_percent(interval=1))
    memory_usage.set(psutil.virtual_memory().percent)
    disk_usage.set(psutil.disk_usage('/').percent)
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded!")

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected!")

        if file:
            # Save the uploaded file
            input_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_file_path)
            #print(f"File saved to {file_path}")

            # Run prediction using YOLO
            results = model.predict(source=input_file_path, save=True)
            print("Prediction results:", results)
            # Get the YOLO directory where results are saved
            # /home/user/runs/detect/predict?
            yolo_save_dir = results[0].save_dir
            #print(f"Results saved to {yolo_save_dir}")
            #print(yolo_save_dir)

            # Move the result image to the results folder
            original_stem = Path(file.filename).stem  
            result_image_name = original_stem + '.jpg'
            result_image_path = os.path.join(yolo_save_dir, result_image_name)
            #print(f"Result image path: {result_image_path}")

            output_file_path = os.path.join(RESULT_FOLDER, os.path.basename(result_image_path))
            #print(f"Moving result image from {result_image_path} to {destination_path}")
            shutil.move(result_image_path, output_file_path)
        
            return render_template('index.html', uploaded_image=input_file_path, result_image=output_file_path)

    return render_template('index.html')

def gen_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        annotated = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/camera')
def camera():
    return Response(gen_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
