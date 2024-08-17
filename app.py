import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


model_custom_cnn = load_model('custom_cnn_model.keras')
model_vgg16 = load_model('vgg16_model_rebuilt.keras')


st.title("Image Classification with Neural Networks")

model_choice = st.radio("Choose a model", ("Custom CNN", "VGG16"), key="model_choice_radio")

if model_choice == "Custom CNN":
    selected_model = model_custom_cnn
elif model_choice == "VGG16":
    selected_model = model_vgg16


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="image_uploader")

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    
    if model_choice == "VGG16":
        image = image.resize((128, 128)) 
    else:
        image = image.resize((28, 28))  
        image = image.convert('L')  

    image_array = np.array(image) / 255.0 

    if model_choice == "VGG16":
        image_array = np.expand_dims(image_array, axis=0)  
    else:
        image_array = np.expand_dims(image_array, axis=-1) 
        image_array = np.expand_dims(image_array, axis=0) 

    
    predictions = selected_model.predict(image_array)
    predicted_class = np.argmax(predictions)

    
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Class probabilities: {predictions}")

