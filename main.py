import streamlit as st
import pandas as pd
import numpy as np
import cv2
from io import BytesIO

from keras.models import load_model
import tensorflow as tf
from tensorflow import keras


st.title('Traffic Sign Board Classification')


from PIL import Image



#st.image(image, caption='Sunrise by the mountains')


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded and display it
if uploaded_image is not None:
    # Display the uploaded image using Streamlit
    

    st.image(uploaded_image, caption="Uploaded Image", use_column_width=False, width=300)

    # Read the image using cv2
    
    img = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert from BGR to RGB
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) 

    pred_class = ['Construction zone sign',
        'do not enter',
        'No passing zone sign',
        'One way sign',
        'Pedestrian crossing sign',
        'railway crossing sign',
        'School zone',
        'Speed limit sign',
        'stop photos',
        'yield sign',
        'miscellaneous']



    model_final = keras.models.load_model('class_model.h5')
    
    # Load and preprocess your image (img) here

    # Make predictions

    predictions = model_final.predict(img)
    class_index = np.argmax(predictions)

    
    pred_class = pred_class[class_index]
    probability = predictions[0][class_index] * 100  # Convert to percentage

    # Display the prediction and probability
    st.subheader("Predicted Class:")
    st.markdown(f'<div style="background-color:#3498db; padding:10px; border-radius:5px;"><h3 style="color:white;">{pred_class}</h3></div>', unsafe_allow_html=True)

    st.subheader("Probability:")
    st.markdown(f'<div style="font-size: 18px; color: #3498db; border: 2px solid #3498db; padding: 8px; border-radius: 5px; font-weight: bold;">{probability:.2f}%</div>', unsafe_allow_html=True)
