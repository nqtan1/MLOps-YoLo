from ultralytics import YOLO
import os
import time
from prometheus_client import Gauge, start_http_server

training_epoch = Gauge('training_epoch', 'Current training epoch')
training_loss = Gauge('training_loss', 'Training loss for the current epoch')
epoch_duration = Gauge('epoch_duration', 'Duration of the current epoch in seconds')

def train_model(model_path: str, data_yaml_path: str, epochs: int = 50, img_size: int = 640) -> None:
    print(f"-> Starting training with YOLO model from {model_path}...")
    model = YOLO(model_path)
    print("-> YOLO model loaded successfully and ready for training.")

    for epoch in range(epochs):
        start_time = time.time()
        print(f"-> Epoch {epoch+1}/{epochs}")
        
        # Update Prometheus metrics for the current epoch
        training_epoch.set(epoch + 1)

        # Train the model for one epoch
        results = model.train(data=data_yaml_path, epochs=1, imgsz=img_size)
        
        # Extract loss from results and update Prometheus
        loss = results.box_loss if hasattr(results, 'box_loss') else 0.0
        training_loss.set(loss)

        # Calculate and update epoch duration
        duration = time.time() - start_time
        epoch_duration.set(duration)

        print(f"-> Training results for epoch {epoch+1}: Loss={loss}, Duration={duration:.2f}s")

    print("-> Training completed successfully.")

def test(model_path: str, data_yaml_path: str, img_size: int = 640) -> None:
    print(f"-> Starting testing with YOLO model from {model_path}...")
    model = YOLO(model_path)
    model.eval()
    print("-> YOLO model loaded successfully and ready for testing.")
    test_result = model.predict(source="data/yolo/images/train/000000000025.jpg"
                                ,conf=0.25
                                ,save=True
                                ,save_txt=True
                                ,imgsz=img_size)
    print(test_result)
    print("-> Testing completed successfully.")

def main():
    model_path = "model/yolov8n.pt"
    data_yaml_path = "./data/yolo/data.yaml"
    epochs = 10
    img_size = 640

    start_http_server(8000)  # Prometheus will scrape metrics from this port
    print("-> Prometheus metrics server started on port 8000.")
    
    train_model(model_path, data_yaml_path, epochs, img_size)
    #test(model_path, data_yaml_path, img_size)

if __name__ == "__main__":
    main()