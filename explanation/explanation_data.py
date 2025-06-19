import subprocess
import os 
import logging

# Logging configuration
logger = logging.getLogger('explanation_data.py')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Data directory structure
IMG_DATA_DIR = "data/yolo"
IMG_TRAIN_DIR = "data/yolo/images/train"
IMG_VAL_DIR = "data/yolo/images/val"
LABEL_TRAIN_DIR = "data/yolo/labels/train"
LABEL_VAL_DIR = "data/yolo/labels/val"

class ExplanationData:
    def __init__(self):
        self.data = {
            'images': {'train': [], 'val': []},
            'labels': {'train': [], 'val': []}
        }
    
    def import_data(self)-> None:
        '''
        Import data from the directory
        '''
        base_dir = ExplanationData.get_directory_path(ExplanationData.get_directory_path())
        logger.info(f"-> Importing data from {os.path.join(base_dir,IMG_DATA_DIR)}...")
        
        self.data['images']['train'] = os.listdir(os.path.join(base_dir, IMG_TRAIN_DIR))
        self.data['images']['val'] = os.listdir(os.path.join(base_dir, IMG_VAL_DIR))
        self.data['labels']['train'] = os.listdir(os.path.join(base_dir, LABEL_TRAIN_DIR))
        self.data['labels']['val'] = os.listdir(os.path.join(base_dir, LABEL_VAL_DIR))

        logger.info(f"-> Data imported successfully from {os.path.join(base_dir,IMG_DATA_DIR)}.")

    def show_explanation (self)-> None:
        '''
        Show the information of YOLO data
        '''
        self.import_data()
        logger.info("-----> YOLO data explanation <-----")
        logger.info(f'1. Data structure:')
        ExplanationData.show_data_structure()
        logger.info(f'2. Data type:')
        ExplanationData.show_data_type(self.data['images']['train'][0], self.data['labels']['train'][0])
        logger.info(f'3. Dataset size (images-labels):')
        ExplanationData.show_num_data(self.data['images']['train'], self.data['images']['val'])
    
    @staticmethod
    def get_directory_path(filePath:str=__file__) -> str:
        return os.path.dirname(os.path.abspath(filePath))
    
    @staticmethod
    def show_data_structure() -> None:
        '''
        Display the structure of YOLO data in the terminal
        '''
        logger.info("YoLo Data Structure:")
        base_dir = ExplanationData.get_directory_path(ExplanationData.get_directory_path())        
        data_dir = os.path.join(base_dir, IMG_DATA_DIR)
        subprocess.run([f'tree -d {data_dir}'],shell=True)
    
    @staticmethod
    def show_data_type(img_file, label_file) -> None:
        '''
        Display the type of YOLO data in the terminal
        '''
        file_name_img, file_extension_img = os.path.splitext(img_file)
        logger.info(f'-> Image file is saved under format: {file_extension_img}')
        
        file_name_label, file_extension_label = os.path.splitext(label_file)
        logger.info(f'-> Label file is saved under format: {file_extension_label}')

    @staticmethod
    def show_num_data(training_set, validation_set) -> None:
        '''
        Display the number of YOLO data in training and validation sets
        '''
        logger.info(f'-> Training set size: {len(training_set)}')
        logger.info(f'-> Validation set size: {len(validation_set)}')
        
if __name__ == "__main__":
    data = ExplanationData()
    data.show_explanation()
