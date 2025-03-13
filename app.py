import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda, Layer
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

def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

class featureExtractionLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = Conv2D(filters=32, kernel_size=(3,3), padding='SAME', activation='relu')
        self.bn = BatchNormalization()
        self.activation = Lambda(hard_tanh)

    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x





st.set_page_config(
    page_title="Synthetic Image Detector",
    layout="wide",
)

# Cache the model loading so it is only loaded once
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            ('syn_classifier.keras'),
            custom_objects={'featureExtractionLayer': featureExtractionLayer}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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
