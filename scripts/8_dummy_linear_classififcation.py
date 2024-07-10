"""
python scripts/8_dummy_linear_image_classification.py --seed 1
python scripts/8_dummy_linear_image_classification.py --seed 2
result is also totally random by the seed
"""

import numpy as np
import cv2
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--seed", required=True, type=int, default=1, help="random seed")
args = vars(ap.parse_args())

# initialize the class labels and set the seed of the pseudorandom number generator so we can reproduce our results
labels = ["dog", "cat", "panda"]
np.random.seed(args["seed"])

#  randomly initialize our weight matrix and bias vector -- in a *real* training and classification task, these parameters would be *learned* by our model, but for the sake of this example, let's use random values
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# load our example image, resize it, and then flatten it into our "feature vector" representation
orig = cv2.imread("../pyimagesearch/data/beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()


# compute the output scores by taking the dot product between the weight matrix and image pixels, followed by adding in the bias
scores = W.dot(image) + b

# loop over the scores + labels and display them
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# draw the label with the highest score on the image as our prediction
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# display our input image
import opencv_jupyter_ui as jcv2
# cv2.imshow("Image", orig)
# cv2.waitKey(0)
cv2.imwrite('output.png', orig)




