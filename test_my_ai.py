"""
test_my_ai.py
Loads a trained MNIST classifier and predicts the digit of a single test image while timing the prediction.
"""

import time
import numpy as np
from PIL import Image   #Used for image processesing.
import joblib           #Used for importing the model.

#Individual test PNG from TestImages.
image_path = "TestImages/image0.png"

#Load the trained model.
clf = joblib.load("number_recognition_model.pkl")

#Load image and convert it to grayscale
img = Image.open(image_path).convert("L")
    
#Resize the image to 28x28
img = img.resize((28, 28))

#Convert to numpy array and flatten
img_array = np.array(img)
img_scaled = img_array.flatten() / 255.0  # scale 0-1

#Invert colors because MNIST data is white digits on black background.
img_scaled = 1.0 - img_scaled

# Get the prediction from the model and time it.
start = time.perf_counter()
prediction = clf.predict([img_scaled])
end = time.perf_counter()

#Get the predicted digit and display results.
digit = prediction[0]
print(f"Predicted digit: {digit}")
print(f"Prediction took {end - start:.6f} seconds")