import streamlit as st
#Importing Required Libraries
import numpy as np
from pickle import dump, load
import pandas as pd
from PIL import Image
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras import Input, layers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from datetime import datetime


st.title('Image Captioning')
#@st.cache
def load_model():
  #Creating Model
  inputs1 = Input(shape=(2048,))
  fe1 = Dropout(0.5)(inputs1)
  fe2 = Dense(256, activation='relu')(fe1)
  inputs2 = Input(shape=(33,))
  se1 = Embedding(1639, 200, mask_zero=True)(inputs2)
  se2 = Dropout(0.5)(se1)
  se3 = LSTM(256)(se2)
  decoder1 = add([fe2, se3])
  decoder2 = Dense(256, activation='relu')(decoder1)
  outputs = Dense(1639, activation='softmax')(decoder2)
  model = Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.load_weights('model_LSTM_20.h5')
  # Load the inception v3 model
  model_img = InceptionV3(weights='imagenet')
  # Create a new model, by removing the last layer (output layer) from the inception v3
  model_new = Model(model_img.input, model_img.layers[-2].output)
  wordtoix = load(open("word2index.pkl", "rb"))
  ixtoword = load(open("index2word.pkl", "rb"))
  return model,model_new,wordtoix,ixtoword

model,model_new,wordtoix,ixtoword=load_model()

#Function to preprocess and resize images
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

# Function to encode a given image into a vector of size (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(33):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=33)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

uploaded_file = st.file_uploader("Choose Image", type="jpg")
now = datetime.now()
a=str(now).replace(' ','_')
a=a.replace('.','_')
a=a.replace(':','_')
a=a.replace('-','_')
zz=a+'.jpg'
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    savee = image.save(zz)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image1=uploaded_file
    image=encode(image1)
    st.text(image)
    image = image.reshape((1,2048))
    b=greedySearch(image)
    st.text(b)  