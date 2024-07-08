"""
python 7_1_your_first_image_classifier.py -d "../pyimagesearch/datasets/animals" -k 1

instead of single neighbor, try k=3 
python 7_1_your_first_image_classifier.py -d "../pyimagesearch/datasets/animals" -k 3
"""

import argparse
import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from imutils.paths import list_images

from pyimagesearch.datasets.simple_datasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

image_paths = list(list_images(args["dataset"]))
preprocessors = [SimplePreprocessor(width=32, height=32)]
dataset_loader = SimpleDatasetLoader(preprocessors)
(data, labels) = dataset_loader.load(image_paths, verbose=500)
data = data.reshape(data.shape[0], -1) # (image_count, feature_vector_length)

# show some information on memory consumption of the images
print("[INFO] features matrix: {:.1f}MB".format(
data.nbytes / (1024 * 1024.0)))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
# -1 jobs for k-NN distance (-1 uses all available cores)
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX)))



