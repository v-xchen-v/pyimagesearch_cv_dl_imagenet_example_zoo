import os
os.environ['KERAS_BACKEND']="torch"

from keras.models import Sequential
from keras import backend as K
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Softmax
from keras.layers import Dense

class ShallowNet:
    """the ShallowNet architecture, contains only a few layers - the entire network architecture can be summerized as: INPUT => CONV => RELU => FC"""
    @staticmethod
    def build(width, height, depth, classes):
        """Initialize the model along with the input shape to be "channel first"""
        model = Sequential()
        input_shape = (height, width, depth) #HWC
        
        # if we are using "channels_first", update the input shape
        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width) #CHW
            
        # define the first (and only) CONV => RELU layer
        model.add(Conv2D(32, (3, 3)), padding="same", input_shape=input_shape)
        model.add(Activation("relu"))
        
        # softmax classifier
        model.add(Flatten())
        model.add(Dense())
        model.add(Activation("softmax"))
        
        return model
        