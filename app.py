import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = r"C:\\Users\\HP\\Desktop\\Project\\potholes.dl\\pothole_detection_model.h5"
model = load_model(MODEL_PATH)

# Set image dimensions
img_height, img_width = 150, 150

# Define a function for image preprocessing
def preprocess_image(image):
    image = image.resize((img_width, img_height))
    image_array = img_to_array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)

# Streamlit app
st.title("Pothole Detection App")
st.write("Upload an image to check if it contains a pothole.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict using the loaded model
    prediction = model.predict(preprocessed_image)
    prediction_value = prediction[0][0]  # Extract scalar value

    # Display the prediction result
    if prediction_value > 0.5:
        st.success("The image contains a pothole.")
    else:
        st.info("The image does not contain a pothole.")

# Add model information
st.sidebar.title("About the Model")
st.sidebar.write("This is a Convolutional Neural Network (CNN) model trained to detect potholes in images.")
st.sidebar.write("The model was trained using TensorFlow and saved as a .h5 file.")
