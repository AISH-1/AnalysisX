import os
import re

import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalMaxPooling1D, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load data from 'customer complaints data.xlsx' into a pandas DataFrame
url = 'customer complaints data.xlsx'
df = pd.read_excel(url, header=0)

# Drop rows with missing values
df.dropna(inplace=True)


# Data Cleaning function to remove unwanted characters from the text
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text)  # Removing @mentions
    text = re.sub(r'#', ' ', text)  # Removing '#' sign
    text = re.sub(r'RT[\s]+', ' ', text)  # Removing RT
    text = re.sub(r'https?\/\/\S+', ' ', text)  # Removing the hyperlinks
    return text


# Apply the cleanText function to the 'TWEET' column of the DataFrame
df['TWEET'] = df['TWEET'].apply(str)
df['TWEET'] = df['TWEET'].apply(cleanText)

# Extract the tweet text and labels as numpy arrays A and B
A = np.array(df['TWEET'])
B = np.array(df['ENCODE'])

# Tokenization and Padding
max_vocab = 2000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(A)

wordix = tokenizer.word_index
V = len(wordix)

train_seq = tokenizer.texts_to_sequences(A)
pad_train = pad_sequences(train_seq)
T = pad_train.shape[1]

D = 20
M = 15

i = Input(shape=(T,))
x = Embedding(V + 1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = GlobalMaxPooling1D()(x)
x = Dense(32, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(i, x)

# Compile the model with optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
r = model.fit(pad_train, B, epochs=50)
