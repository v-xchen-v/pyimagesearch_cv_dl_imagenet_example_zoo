from .base_preprocessor import Preprocessor
from keras.api.utils import img_to_array

class ImageToArrayPreprocessor(Preprocessor):
    """an image preprocessor wraps keras' img_to_array function that accepts an input image then properly orders the channel based on our image_data_format setting"""
    
    def __init__(self, data_format=None) -> None:
        super().__init__()
        # store the data format
        self.data_format = data_format
        
        
    def preprocess(self, image):
        # apply the Keras utility function that correctly rearranges the dimensions of the image
        return img_to_array(image, data_format=self.data_format)
        