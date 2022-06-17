# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 10:18:03 2022

@author: Acer
"""

import pandas as pd
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional,Embedding

#%%
URL_PATH = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

# EDA
# DATA LOADING
df = pd.read_csv(URL_PATH)
vocab_size = 10000
oov_token = 'OOV'
max_len = 180

#%%
# df_copy = df.copy() # backup
# df = df_copy.copy() # check point

#%%
# DATA INSPECTION
df.head(10)
df.tail(10)
df.info()

df['sentiment'].unique() # to get the unique target
df['review'][5]
df['sentiment'][5]
df.duplicated().sum() # 418 duplicated data
df[df.duplicated()]

# <br /> HTML tags that have to be removed
# numbers can be filtered
# need to remove duplicated data

# DATA CLEANING
df = df.drop_duplicates() # remove duplicate data

# remove HTML tags
# '<br /> bdjsbdebdck <br />'.replace('<br />',' ') --> example
# df['review'][10].replace('<br />',' ')

review = df['review'].values # features X
sentiment = df['sentiment'].values # target y

# better way
for index,rev in enumerate(review):
    # remove html tags
    # ? dont be greedy
    # * zero or more occurances
    # any character except new line(/n)
    review[index] = re.sub('<.*?>',' ',rev)
    
    # convert into lower case
    # remove numbers
    # ^ means NOT
    review[index] = re.sub('[^a-zA-Z]',' ', rev).lower().split()


# FEATURES SELECTION
# nothing to select

#%%
# PREPROCESSING (convert into lower case, 
#tokenization, padding and truncating, one hot encoding )
# tokenization

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(review)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(review) # to convert into numbers

# padding and truncating
length_of_review = [len(i) for i in train_sequences] # list comprehension
print(np.median(length_of_review)) # to get the number of max length for padding

padded_review = pad_sequences(train_sequences,maxlen=max_len,
                              padding='post',truncating='post')

# One hot encoding
ohe = OneHotEncoder(sparse=False)
sentiment = ohe.fit_transform(np.expand_dims(sentiment,axis=-1))

# train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_review,
                                                 sentiment,
                                                 test_size=0.3,
                                                 random_state=123)

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model development
# using LSTM layers, dropout, dense, input
# achieved 90% f1 score

embedding_dim = 64
model = Sequential()
model.add(Input(shape=(180))) # or np.shape(X_train)[1:]
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(embedding_dim,return_sequences=(True)))) # bidirectional
# model.add(LSTM(128,return_sequences=(True)))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.summary()

plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

hist = model.fit(X_train,y_train,batch_size=128,epochs=5,
                 validation_data=(X_test,y_test))

#%%
hist.history.keys()

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training loss','validation loss'])
plt.title('Loss')
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training accuracy','validation accuracy'])
plt.title('Accuracy')
plt.show()

#%% Model evaluation

y_true = np.argmax(y_test,axis=1)
y_pred = np.argmax(model.predict(X_test),axis=1)

cr = classification_report(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
acc_score = accuracy_score(y_true, y_pred)

print(cr)
print(cm)
print(acc_score)

#%% Model saving
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_sentiment.json')
with open(TOKENIZER_PATH,'w') as json_file:
    json.dump(token_json,json_file)

OHE_PATH = os.path.join(os.getcwd(),'ohe_pkl')
with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
    
#%% Discussion/reporting
'''
Discuss the result:
    Model achieved 85% accuracy during training. Recall and f1 score reports
    81% and 84% respectively. However, the model starts to overfit after 2nd
    epochs.
    Early stopping can be introduced to prevent overfitting.
    Increase dropout data to control overfitting.
    
Write about model architechture:
    Trying with different architechture. Eg; BERT model, transformer model,
    GPT3 model my help to improve the model

1. result
2. give suggestion
3. gather evidence showing what went wrong during training/model development
'''






