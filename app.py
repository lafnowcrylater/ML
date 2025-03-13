import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from filters import apply_all_filters
from s_and_r import smash_n_reconstruct

def process(image):
    rich, poor = smash_n_reconstruct(image)
    rich = tf.cast(tf.expand_dims(apply_all_filters(rich), axis=-1), dtype=tf.float32)
    poor = tf.cast(tf.expand_dims(apply_all_filters(poor), axis=-1), dtype=tf.float32)

    rich = rich / 255.0
    poor = poor / 255.0

    rich.set_shape([100, 100, 1])
    poor.set_shape([100, 100, 1])

    return rich, poor





# Set up the page configuration
st.set_page_config(
    page_title="Synthetic Image Detector",
    layout="wide",
)

# Cache the model loading so it is only loaded once
@st.cache(allow_output_mutation=True)
def load_model():
    # Update the model path as needed
    model = tf.keras.models.load_model("syn_classifier.keras")
    return model

model = load_model()

st.title("Image Upload and Prediction App")

# File uploader accepts jpg, jpeg, and png files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing image...")
    # Preprocess the image: resize, convert to array, normalize, etc.
    target_size = (200, 200)  # change this to match your model's input size
    image = ImageOps.fit(image, target_size, Image.NEAREST)

    rich, poor = process(image)
    
    # Get prediction from the model
    predictions = model.predict([rich, poor])

    # Decide the class based on a threshold (0.5 is a common choice)
    predicted_prob = predictions[0][0]  # assuming model output shape is (1, 1)
    if predicted_prob >= 0.5:
        predicted_class = "Fake"  # or your class label for 1
    else:
        predicted_class = "Real"  # or your class label for 0

    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {predicted_prob:.2f}" if predicted_class == "Positive" else f"**Confidence:** {1 - predicted_prob:.2f}")
