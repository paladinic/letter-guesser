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



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("static/css/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')






# UI --------

with col1:
    st.write("""
        ## Letter Guesser
        """)
    canvas_result = st_canvas(stroke_width=25,width=250,height=250)

with col2:
    if canvas_result.image_data is not None:
        img = adjust_img(canvas_result.image_data)
        pred = model.predict(img)

        res = pd.DataFrame({'pred': pred.reshape(-1), 'letter' : alphabet})
        res['max'] = res.pred == max(res.pred)
        fig = px.bar(res,x='letter',y='pred',color='max')
        fig.update_layout(xaxis={'categoryorder':'category ascending'})
        st.plotly_chart(fig)
