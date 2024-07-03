import cv2
from .base_preprocessor import Preprocessor

class SimplePreprocessor(Preprocessor):
    """
    an image preprocessor that resizes the image, ignoring the aspect ratio.
    """
    
    def __init__(self, width, height, inter=cv2.INTER_AREA) -> None:
        """store the target image width, height and interpolation method used when resizing
        """
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        """resize the image to a fixed size, ignoring the aspect ratio"""
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)