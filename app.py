from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import requests

st.write('# MNIST Digit Recognition')
st.write('## Using a CNN `PyTorch` model')


st.write('### Draw a digit in 0-9 in the box below')
# Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 18)

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    resp = requests.get("http://localhost:8383/", json={"array": test_x.tolist()})

    st.write('### Prediction') 
    st.write('### '+ resp.text)