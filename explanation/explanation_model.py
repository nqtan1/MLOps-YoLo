from ultralytics import YOLO
import os 
import logging  

# Logging configuration
logger = logging.getLogger('explanation_model.py')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class YOLOModelExplanation:
    def __init__(self, model_path:str) -> None:
        '''
        Initialize the YOLO model for explanation
        '''
        self.model_path = model_path
        self.model = YOLO(model_path)

    def show_model_info(self) -> None:
        '''
        Show the information of YOLO model from yaml file
        ''' 
        config = self.model.model.yaml

        logging.info("-----> YOLO model explanation <-----") 
        logging.info(f'1. Model name: {self.get_model_name_from_path(self.model_path)}')
        logging.info(f'2. Number of classes: {config['nc']}')
        logging.info(f'3. Input channel: {config['ch']}')
        logging.info(f'4. Depth multiple: {config['depth_multiple']}')
        logging.info(f'5. Width multiple: {config['width_multiple']}')
        logging.info(f'6. Backbone structure: {config['backbone']}')
        logging.info(f'7. Head: {config['head']}')

    @staticmethod
    def get_directory_path(filePath:str=__file__) -> str:
        return os.path.dirname(os.path.abspath(filePath))

    @staticmethod
    def get_base_dir() -> str:
        '''
        Get the base directory of the model
        '''
        return os.path.dirname(YOLOModelExplanation.get_directory_path())
    
    @staticmethod
    def get_model_name_from_path(model_path: str) -> str:
        return os.path.splitext(os.path.basename(model_path))[0]

def main():
    model_path = YOLOModelExplanation.get_base_dir() + "/model/yolov8n.pt"
    yolo_model_explanation = YOLOModelExplanation(model_path)
    yolo_model_explanation.show_model_info()

if __name__ == "__main__":
    main()