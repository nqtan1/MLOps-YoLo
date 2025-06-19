import os 
import shutil
import requests
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import logging

# Logging configuration
logger = logging.getLogger('data.py')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def unzip_data(zip_file_path: str, output_dir: str) -> None:
    # Unzip the file if it's a zip file 
    if zip_file_path.endswith(".zip"):
        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"-> Unzipped data to {output_dir}.")
        os.remove(zip_file_path)  # Remove the zip file after extraction
    else:
        logger.info(f"-> Data is not a zip file. No extraction needed.")

def download_data(url: str, output_dir: str,filename: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    logger.info(f"-> Downloading data from {url}...") 
    response = requests.get(url,stream=True)

    with open(file_path,"wb") as file:
        # Write the response content to a file in chunks
        for chunk in response.iter_content(chunk_size = 1024): # 1KB chunks to avoid memory issues
            if chunk:
                file.write(chunk)

    logger.info(f"-> Data downloaded to {file_path}.")

    unzip_data(file_path, output_dir)

def create_yolo_structure(base_dir: str) -> None:
    # Create the YOLO directory structure 
    directories = [
        os.path.join(base_dir, "images/train"),
        os.path.join(base_dir, "images/val"),
        os.path.join(base_dir, "labels/train"),
        os.path.join(base_dir, "labels/val")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"-> YoLo data structure created at {base_dir}.")

def clean_data(base_dir: str) -> None:
    # Remove all data in the base directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        logger.info(f"-> Cleaned data in {base_dir}.")
    else:
        logger.info(f"-> {base_dir} does not exist. No cleaning needed.")

def check_yolo_structure(base_dir: str) -> None:
    # Check if the YOLO directory structure exists
    directories = [
        os.path.join(base_dir, "images/train"),
        os.path.join(base_dir, "images/val"),
        os.path.join(base_dir, "labels/train"),
        os.path.join(base_dir, "labels/val")
    ]

    for directory in directories:
        if not os.path.exists(directory):
            logger.info(f"-> {directory} does not exist.")
            return
    logger.info(f"-> YOLO data structure is valid at {base_dir}.")

def split_data(source_dir:str, destination_dir:str, test_size:float = 0.2) -> None:

    img_dir = os.path.join(source_dir, "images/train2017")
    label_dir = os.path.join(source_dir, "labels/train2017")

    # Split the data into training and validation sets 
    images = [img for img in os.listdir(img_dir) if img.endswith(('.jpg', '.png'))]
    labels = [label for label in os.listdir(label_dir) if label.endswith('.txt')]

    # Ensure that the number of images and labels are the same
    images = [img for img in images if img.replace('.jpg', '.txt') in labels]

    assert len(images) > 0, "No images found in the source directory."
    assert len(labels) > 0, "No labels found in the source directory."

    # Split the data into training and validation sets
    train_images, val_images = train_test_split(images
                                                ,test_size=test_size
                                                ,random_state=42)

    for img in train_images:
        shutil.copy(os.path.join(img_dir, img)
                    ,os.path.join(destination_dir
                    , "images/train", img)) 
        shutil.copy(os.path.join(label_dir, img.replace('.jpg', '.txt')).replace('.png','txt')
                                                        , os.path.join(destination_dir, "labels/train"
                                                        , img.replace('.jpg', '.txt').replace('.png','txt')))

    for img in val_images:
        shutil.copy(os.path.join(img_dir, img)
                    ,os.path.join(destination_dir, "images/val", img)) 
        shutil.copy(os.path.join(label_dir, img.replace('.jpg', '.txt')).replace('.png','txt')
                    ,os.path.join(destination_dir, "labels/val", img.replace('.jpg', '.txt').replace('.png','txt')))
    logger.info(f"-> Data split into training and validation sets in {destination_dir}.")

def main() -> None:
    dataSet_url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    output_dir = "data"
    filename = "coco128.zip"
    download_data(dataSet_url, output_dir, filename)
    create_yolo_structure(base_dir='data/yolo')
    check_yolo_structure(base_dir='data/yolo')
    split_data(source_dir='data/coco128', destination_dir='data/yolo', test_size=0.2)

if __name__ == "__main__":
    main()