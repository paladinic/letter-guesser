# LIBS --------

# streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# data
import numpy as np
import pandas as pd

# moddelling
from keras.models import load_model

# visualisation
import plotly.express as px

# images
import cv2

# LAYOUT --------
st.set_page_config(layout="wide")
col1, col2 = st.columns((1,2))



# MODEL -------

# upload model
model = load_model('static/models/letters_model.h5py')

# udf's
alphabet = np.char.upper(np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]))
def alph_res(pred):
    return alphabet[pred == max(pred)][0]

def adjust_img(img):
    resizedImage = cv2.resize(img.astype('float32'), (28,28))[:,:,3]
    rescaledImage = resizedImage*(16/255)
    reshapedImage = rescaledImage.reshape(1,784)
    return reshapedImage



# UI --------

st.write("""
    # Letter Guesser
    Simple letter guessing interface with *keras* and *streamlit.io*
""")


with col1:
    canvas_result = st_canvas(width=250,height=250)

if canvas_result.image_data is not None:
    img = adjust_img(canvas_result.image_data)
    pred = model.predict(img)
    res = pd.DataFrame({'pred': pred.reshape(-1), 'letter' : alphabet})
    fig = px.bar(res,x='letter',y='pred')
    with col2: st.plotly_chart(fig)
