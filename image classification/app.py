import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from PIL import Image

# Load the trained model
model = load_model('cifar10_cnn_model.h5')

# Define class labels
class_labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Function to preprocess and predict an image
def predict_image(img, model, class_labels):
    # Resize and normalize the image
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class, prediction

# Function to load a local image and convert it to base64
def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Load the background image from the given local path
background_image_path = r"C:\Users\mayur\Desktop\image classification\back.webp"
background_image_base64 = load_image(background_image_path)

# Set background style and dark text color using base64-encoded image
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/webp;base64,{background_image_base64}) no-repeat center center fixed;
        background-size: cover;
    }}
    h1, h2, h3, h4, h5, h6, p {{
        color: #333333;  /* Dark color for text */
    }}
    .stButton>button {{
        color: white;
        background-color: #4CAF50;  /* Button background color */
    }}
    .predictions-table {{
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(0, 0, 0, 0.5);
        color: white;
    }}
    .predicted-class {{
        font-weight: bold;
        font-size: 20px;
        color: yellow;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app UI
st.title("Image Classification with CIFAR-10 CNN")
st.write("Upload an image to classify it into one of the CIFAR-10 categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Predict the image
    with st.spinner("Classifying..."):
        predicted_class, prediction = predict_image(img, model, class_labels)

    # Display the predicted class with better styling
    st.markdown(f'<p class="predicted-class">Predicted Class: {predicted_class}</p>', unsafe_allow_html=True)

    # Display the prediction probabilities in a table format
    st.markdown('<div class="predictions-table">', unsafe_allow_html=True)
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob * 100:.2f}%")
    st.markdown('</div>', unsafe_allow_html=True)
