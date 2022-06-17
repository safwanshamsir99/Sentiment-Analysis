# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 11:51:59 2022

@author: Acer
"""
from tensorflow.keras.models import load_model
import os
import json
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
# path
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
OHE_PATH = os.path.join(os.getcwd(),'ohe_pkl')

# to load the model
loaded_model = load_model(os.path.join(os.getcwd(),'model.h5'))

loaded_model.summary()

# to load tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%%
# input_review = 'The movie is bad. Wasted my time.'
input_review = input('Type your review here: ')

# preprocessing
input_review = re.sub('<.*?>',' ',input_review)
input_review = re.sub('[^a-zA-Z]',' ', input_review).lower().split()

tokenizer = tokenizer_from_json(loaded_tokenizer)
input_review_encoded = tokenizer.texts_to_sequences(input_review)

input_review_encoded = pad_sequences(np.array(input_review_encoded).T,
                                     maxlen=180,padding='post',
                                     truncating='post')

outcome = loaded_model.predict(np.expand_dims(input_review_encoded,axis=-1))

with open(OHE_PATH,'rb') as file:
    loaded_ohe = pickle.load(file)
    
print(loaded_ohe.inverse_transform(outcome))