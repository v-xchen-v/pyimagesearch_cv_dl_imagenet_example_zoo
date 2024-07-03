import numpy as np
import cv2
import os
from typing import List
from ..preprocessing import simple_preprocessor
from ..preprocessing.base_preprocessor import Preprocessor

class SimpleDatasetLoader:
    def __init__(self, preprocessors: List[Preprocessor]=None):
        # store the image preprocessor
        self.preprocessors = preprocessors
        
        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, image_paths: List[str], verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        
        # loop over the input images
        for (i, image_path) in enumerate(image_paths):
            """Load the image and extract the class label assuming that our path has the following format:
            /path/to/dataset/{class}/{image}.jpg"""
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]
            
            # loop over the preprocessors and apply each to the image
            for p in self.preprocessors:
                image = p.preprocess
                
            # treat our processed image as a "feature vector" by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            
            # show an update every 'verbose' image
            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print(f"[INFO] processed {i+1}/{len(image_paths)}")
            
        # return a tuple of the data and label
        return (np.array(data), np.array(labels))