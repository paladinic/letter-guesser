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
nn_model = load_model('static/models/letters_model.h5py')
cnn_model = load_model('static/models/conv_model.h5')

# udf's
alphabet = np.char.upper(np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]))
def alph_res(pred):
    return alphabet[pred == max(pred)][0]

def adjust_img_2d(img):
    resizedImage = cv2.resize(img.astype('float32'), (28,28))[:,:,3]
    rescaledImage = resizedImage*(16/255)
    reshapedImage = rescaledImage.reshape(-1, 28,28, 1)
    return reshapedImage

def adjust_img_1d(img):
    resizedImage = cv2.resize(img.astype('float32'), (28,28))[:,:,3]
    rescaledImage = resizedImage*(16/255)
    reshapedImage = rescaledImage.reshape(1,784)
    return reshapedImage

# UI --------

with col1:
    st.write("""
        # Letter Guesser
        [*Claudio Paladini*](www.paladinic.com)
        """)
    canvas_result = st_canvas(stroke_width=30,width=250,height=250)

with col2:
    if canvas_result.image_data is not None:

        # img1d = adjust_img_1d(canvas_result.image_data)
        # pred_nn = nn_model.predict(img1d)
        # res_nn = pd.DataFrame({'Prediction': pred_nn.reshape(-1), 'letter' : alphabet})
        # res_nn['Max'] = res_nn.Prediction == max(res_nn.Prediction)
        # fig = px.bar(res_nn,x='letter',y='Prediction',color='Max')


        img2d = adjust_img_2d(canvas_result.image_data)
        pred_cnn = cnn_model.predict(img2d)
        res_cnn = pd.DataFrame({'Prediction': pred_cnn.reshape(-1), 'letter' : alphabet})
        res_cnn['Max'] = res_cnn.Prediction == max(res_cnn.Prediction)
        fig = px.bar(res_cnn,x='letter',y='Prediction',color='Max')

        fig.update_layout(xaxis={'categoryorder':'category ascending'})
        st.plotly_chart(fig)
