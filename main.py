"""
main.py

Main script to train and evaluate a support vector machine (SVM) for MNIST digit recognition.

Workflow:
1. Load MNIST training and test datasets from binary files.
2. Scale pixel values to 0-1 range.
3. Split training data into a smaller subset for quicker training.
4. Train an SVM classifier on the subset.
5. Evaluate predictions and print metrics.
6. Optionally display sample predictions.
7. Save the trained model to disk.

Dependencies:
- numpy
- matplotlib
- scikit-learn
- joblib
"""

import numpy as np
import matplotlib.pyplot as plt

#Import a second script containing necessary functions.
import training_tools

from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

display_image_amount = 14   #Set an amount of images to display if training_tools.display() is used.
random_images = False       #If training_tools.display() is used, it randomized the images that appear.
clf_image_amount = 1000     #Set the amount of images clf is meant to process for testing purposes.

#Paths to the desired training and testing images/labels are required to be defined.
train_images = ("MNISTTrainingDataSet/train-images.idx3-ubyte") #Path to training images.
train_labels = ("MNISTTrainingDataSet/train-labels.idx1-ubyte") #Path to training labels.
test_images = ("MNISTTrainingDataSet/t10k-images.idx3-ubyte")   #Path to testing images.
test_labels = ("MNISTTrainingDataSet/t10k-labels.idx1-ubyte")   #Path to testing labels.

#Only run if this is the main function.
def main():

    #Let the user know the program has started.
    print("Activated.")

    #Load data sets and convert them from binary to a uint8, a more usable pixel format.
    X_train = training_tools.load_mnist_images(train_images)
    Y_train = training_tools.load_mnist_labels(train_labels)
    X_test  = training_tools.load_mnist_images(test_images)
    Y_test  = training_tools.load_mnist_labels(test_labels)

    # Create the model to used a Support Vector Machine (SVM) using scikit-learn's implementation to create a Support Vector Classifier (SVC) model.
    clf = svm.SVC(gamma=0.035) #Low gamma value to ensure model does not have too much classification overlap. This was determined with testing.

    #Decrease pixel values for easier AI readability
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0

    #Tell the algorithm how to train the model using train_test_split.                                                      Train_size = the percentage of the training set that the algorithm will use to train.
    print("Train_test_split...")                                                                                            #random_state allows multiple scenarios with a set int for reproducable results.
    X_small, _, Y_small, _ = train_test_split(X_train_scaled, Y_train, train_size=0.05, random_state=3, stratify=Y_train)   #stratify = Y_train to ensure proper balance between testing results
                                                                                                                            #   -> use Y_train because that contains the actual image labels.

    #Here is where the model begins to train through clf.
    print("Classifying.")
    clf.fit(X_small, Y_small)

    #Prediction allows clf to compare model performance with the given clf_image_amount test images.
    predicted = clf.predict(X_test_scaled[:clf_image_amount])

    #Below are possible functions that could be used to illustrate model performance.

    #training_tools.predict(Y_test, clf_image_amount, predicted)                                         #Display prediction statistics about the model.
    #training_tools.score(clf, X_test_scaled,Y_test,clf_image_amount)                                    #Display a score for the model with the given amount of clf images.
    #training_tools.display(X_test, predicted, display_image_amount, clf_image_amount, random_images=1)  #Display a certain amount of images, correlating to display_image_amount.
    training_tools.confusion_map(Y_test, predicted, clf)                                                #Display a confusion matrix heatmap to demonstrate accuracy.

    #training_tools.save_model(clf)     #This function allows the user to save the model if so desired.

    #Display that the program has finished running.
    print("Finished.")

if __name__ == "__main__":
    main()