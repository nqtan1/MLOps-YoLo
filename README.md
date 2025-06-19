# MLOps-YoLo

This project is a simple web-based application for object detection using YOLOv8n and PyTorch. It provides a user-friendly interface for uploading images and visualizing detection results using a trained YOLOv8n model.

## 🚀 Features

- 🔍 Object detection using YOLOv8 (PyTorch version)
- 📷 Upload and process custom images
- 🌐 Flask-based web interface
- 💾 Pretrained model included (`yolov8n.pt`)
- 📊 Modular code for easy training and inference

## 📁 Project Structure



MLOps-YoLo/
│
├── app.py                 # Main Flask application
├── train.py               # Training script for YOLOv8
├── yolov8n.pt             # Pretrained model file (not tracked by GitHub)
│
├── static/
│   └── uploads/           # Uploaded images
│
├── templates/
│   └── index.html         # HTML template for the web interface
│
├── requirements.txt       # Python dependencies
└── README.md              # Project description (this file)



## ⚙️ Installation

```bash
git clone https://github.com/nqtan1/MLOps-YoLo.git
cd MLOps-YoLo

# Create virtual environment (optional)
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt


## 🧪 Usage

### 1. Run the Web Application

```bash
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 2. Train the YOLOv8 Model

You can modify and run the training script:

```bash
python train.py
```

> Make sure you have your dataset in the correct format (YOLO format) and update the paths inside `train.py`.

## 💡 Notes

* The pretrained file `yolov8n.pt` exceeds 100MB and is **not pushed to GitHub** due to file size limitations. You should download it manually from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
* You can also use Git LFS if you plan to manage large model files.

## 📌 TODO

* [ ] Add support for video stream
* [ ] Integrate model evaluation metrics
* [ ] Add Dockerfile for deployment


