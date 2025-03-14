import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Lambda, Layer
import numpy as np
from PIL import Image
import PIL
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

st.title("Synthetic Image Detector")

# File uploader accepts jpg, jpeg, and png files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    st.write("Processing image...")

    rich, poor = process(image)

    rich_batch = np.expand_dims(rich, axis=0)  # Shape becomes (1, 100, 100, 1)
    poor_batch = np.expand_dims(poor, axis=0)  # Shape becomes (1, 100, 100, 1)

    # Get prediction from the model
    predictions = model.predict([rich_batch, poor_batch])

    # Decide the class based on a threshold (0.5 is a common choice)
    predicted_prob = predictions[0][0]  # assuming model output shape is (1, 1)
    if predicted_prob >= 0.23:
        predicted_class = "Fake"  # class label for 1
    else:
        predicted_class = "Real"  # class label for 0

    st.header("Prediction Results")
    st.subheader(f"**Predicted Class:** {predicted_class}")
    #st.write(f"**Confidence:** {predicted_prob:.2f}" if predicted_class == "Positive" else f"**Confidence:** {1 - predicted_prob:.2f}")
