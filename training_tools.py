"""
training_tools.py

Utility functions for training, evaluating, and displaying results
from an MNIST digit recognition model.

Functions:
- load_mnist_images(filename): Load MNIST images from a binary file.
- load_mnist_labels(filename): Load MNIST labels from a binary file.
- display(X, Y, image_amount, clf_image_amount, random_images): Display images with their labels/predictions.
- predict(Y_test, clf_image_amount, predicted): Print classification report for predictions.
- confusion_map(Y_test, Y_predict, clf): Display predicted versus true labels in a heatmap.
- score(clf, X_test, Y_test, clf_image_amount): Print the accuracy score of a trained model.
- save_model(clf): Save a trained classifier to disk using joblib.
"""

import numpy as np
import struct                   #Used for unpacking binary files.
import random
import matplotlib.pyplot as plt #Used for plotting results.
import seaborn as sns           #Used for generating heatmaps
import joblib                   #Used for saving the model.

from sklearn import datasets, metrics, svm              #Library used for model training.
from sklearn.model_selection import train_test_split    #The train_test_split is used for training the model.
from sklearn.metrics import confusion_matrix

#Unpack images from binary into uint8.
def load_mnist_images(filename):
    """
    Reads MNIST images from a binary file.

    Args:
        filename (str): Path to the .idx3-ubyte image file.

    Returns:
        np.ndarray: Array of shape (num_images, 784) with dtype uint8.
    """

    with open(filename, 'rb') as f: #Read the file with a binary interpretation.

        magic, num, rows, cols = struct.unpack(">IIII", f.read(16)) #Separate the header from 16 bytes into four groups of 4 bytes
        #Magic: file type, Num: number of images in the file, Rows: number of rows in the file, Cols: number of columns in the file

        images = np.frombuffer(f.read(), dtype=np.uint8) #All remaining data is stored into images as uint8 bytes, AKA: pixel values
        images = images.reshape(num, 28*28) #This separates the images from one giant array to multiple image arrays, separating images every 784 indexes (28x28)

    return images

#Unpack labels from binary into uint8.
def load_mnist_labels(filename):
    """
    Reads MNIST labels from a binary file.

    Args:
        filename (str): Path to the .idx1-ubyte label file.

    Returns:
        np.ndarray: Array of shape (num_labels,) containing labels as uint8. Each label is an integer between 0 and 9.
    """

    with open(filename, 'rb') as f: #Read the file with a binary interpretation

        magic, num = struct.unpack(">II", f.read(8)) #Separate the header 8 bytes into two groups of 4 bytes
        #Magic: file type, Num: number of labels in the file

        labels = np.frombuffer(f.read(), dtype=np.uint8) #Read the remaining bytes as labels, 
    return labels

#Display images in a plot.
def display(X, Y, image_amount, clf_image_amount, random_images):
    """
    Displays images in a plot.
    
    Args: 
        X (np.ndarray): Array of flattened images (shape: num_images x 784).
        Y (np.ndarray): Array of labels or predictions corresponding to X.
        image_amount (int): Number of images to display.
        clf_image_amount (int): Number of accessible test images.
        random_images (bool): If True, images are selected randomly from X; 
                              if False, the first `image_amount` images are used.

    Returns:
        None
    """

    if random_images:
        X = X.copy()
        Y = Y.copy()

        for i in range(image_amount):
            random_index = random.randint(0,clf_image_amount - (1 + image_amount))
            X[i] = X[random_index]
            Y[i] = Y[random_index]
    
    print("Plotting.")

    rows = 2                        #Always have 2 rows
    columns = image_amount//rows    #Columns will increase by 2, depending on image amount.

    _, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 3))  #Use plt to create a grid with corresponding rows and columns. Figsize determines plot GUI dimensions.
    axes = axes.flatten()                                               #Flatten the 2D array into a 1D array for variable image amount.

    #Zip takes the properties from axes, and groups X and Y indexes together
    for ax, image, prediction in zip(axes, X[:image_amount], Y[:image_amount]):

        ax.set_axis_off()                                               #Clean the image by removing traditional graph lines.
        image = image.reshape(28, 28)                                   #Images are 1D (784 length), so turn them back to 28x28.
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")   #Display the images in reverse gray scale (reflects MNIST data).
                                                                        #Interpolation ensures sharp pixel edge.
        ax.set_title(f"Prediction: {prediction}")                       #Label each plot with its predicted label.

    plt.show()  #Show the plot!

#Get prediction values
def predict(Y_test, clf_image_amount, predicted):
    """
    Prints classification report comparing labels to images.
    
    Args:
        Y_test (np.ndarray): Array of true labels for the test set.
        clf_image_amount (int): Number of test images to include in the report.
        predicted (np.ndarray): Array of predicted labels from the classifier.

    Returns:
        None
    """
    print("Predicting.")
    print(metrics.classification_report(Y_test[:clf_image_amount], predicted))  #Get certain amount of predictions based on clf_image_amount

def confusion_map(Y_test, Y_predict, clf):
    """
    Display confusion_matrix in a heatmap.
    
    Args:
        Y_test (np.ndarray): Array of true labels from the test set.
        Y_predict (np.ndarray): Array of prediction labels from the model.
        clf (class): Classifier of the model.

    Returns:
        None
    """
    print("Creating heatmap.")

    Y_test = Y_test[:Y_predict.shape[0]]    #Reshape Y_test so it always equals same bounds of Y_predict.
    cm = confusion_matrix(Y_test, Y_predict, labels=clf.classes_)   #Create a confusion matrix.

    #Establish configs of the heatmap.
    plt.figure(figsize=(10,7))

    #Set the heatmap within the plot.
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=clf.classes_, yticklabels=clf.classes_)

    #Create descriptive labels.
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix Heatmap")

    plt.show()

#Score the model
def score(clf, X_test, Y_test, clf_image_amount):
    """
    Prints the accuracy score of a trained classifier on a subset of test data.

    Args:
        clf (sklearn.svm.SVC or similar): Trained classifier object.
        X_test (np.ndarray): Array of test images (flattened and scaled).
        Y_test (np.ndarray): Array of true labels corresponding to X_test.
        clf_image_amount (int): Number of test images to include in the score calculation.

    Returns:
        None
    """

    print("Score: " + str(clf.score(X_test[:clf_image_amount], Y_test[:clf_image_amount]))) #Scores the model: testing images against testing labels.

#Save the model
def save_model(clf):
    """
    Saves a trained classifier to disk using joblib.

    Args:
        clf (sklearn.svm.SVC or similar): Trained classifier object to save.

    Returns:
        None

    Side Effects:
        Creates a file named "number_recognition_model.pkl" in the current directory.
    """

    print("Saving model.")
    joblib.dump(clf, "number_recognition_model.pkl") #Save the model in a file called "number_recognition_model.pkl"
    print("Saved.")