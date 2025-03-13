import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Set up the page configuration
st.set_page_config(
    page_title="Image Prediction App",
    layout="wide",
)

# Cache the model loading so it is only loaded once
@st.cache(allow_output_mutation=True)
def load_model():
    # Update the model path as needed
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Define your class names (modify these to match your model)
class_names = ["class1", "class2", "class3"]

st.title("Image Upload and Prediction App")

# File uploader accepts jpg, jpeg, and png files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing image...")
    # Preprocess the image: resize, convert to array, normalize, etc.
    target_size = (224, 224)  # change this to match your model's input size
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    img_array = np.asarray(image)
    # Normalize if the model expects float inputs (adjust if necessary)
    img_array = img_array.astype("float32") / 255.0
    # Expand dimensions to match model's input shape (batch size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get predictions from the model
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]
    
    st.write("### Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
