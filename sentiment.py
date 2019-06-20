#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn import metrics
import nltk
import os
import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout,Embedding,LSTM, CuDNNGRU, Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,GlobalMaxPool1D,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")


# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


punctuation = string.punctuation

stopword = stopwords.words("english")

lem = WordNetLemmatizer()

def clean(text):
    
    text = text.lower()
    
    text = re.sub(r"http\S+", "", text)
    
    # punctuation removal
    text = "".join(p for p in text if p not in punctuation)
    
    # stopwords removal
    words = text.split()
    words = [w for w in words if w not in stopword]
    
    # lemitization
    words = [lem.lemmatize(word,'v') for word in words]
    words = [lem.lemmatize(word,'n') for word in words]
    
    text = " ".join(words)
    
    return text

#checking the function
clean('This is a test for checking if function is working properly? https//:www.google.com')
    
    


# In[6]:


train['cleaned'] = train['tweet'].apply(clean)
test['cleaned'] = test['tweet'].apply(clean)


# In[7]:


train.head()


# In[8]:


target = to_categorical(train['label'])
train = train.drop('label', axis = 1)


# In[9]:


x_train, x_val, y_train, y_val = train_test_split(train['cleaned'], target, test_size = 0.2, random_state = 1)


# In[10]:


words = " ".join(x_train)
words = nltk.word_tokenize(words)
dist = nltk.FreqDist(words)
num_unique_words = len(dist)


# In[11]:


r_len = []
for w in x_train:
    word=nltk.word_tokenize(w)
    l=len(word)
    r_len.append(l)
max_len = np.max(r_len)
max_len


# In[12]:


max_features = num_unique_words
max_words = max_len
batch_size = 128
embed_dim = 300


# In[13]:


tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test  = tokenizer.texts_to_sequences(test['cleaned'])


# In[14]:


x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_val = sequence.pad_sequences(x_val, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)
print(x_train.shape, x_val.shape, x_test.shape)


# ## GRU model

# In[15]:


inp = Input(shape=(max_words,))
x = Embedding(max_features, embed_dim)(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(2, activation = "softmax")(x)
model1 = Model(inputs = inp, outputs=x)
model1.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model1.summary())


# In[16]:


model1.fit(x_train, y_train, batch_size=512, epochs=20, validation_data=(x_val, y_val))


# In[17]:


pred1 = np.round(np.clip(model1.predict(x_val),  0, 1))
print(f1_score(y_val, pred1, average=None))


# In[18]:


pred1=np.round(np.clip(model1.predict(x_test), 0, 1)).astype(int)
pred1 = pd.DataFrame(pred1)
pred1 = pred1.idxmax(axis=1)
submission_GRU = pd.DataFrame({'id':test['id'], 'label':pred1})
submission_GRU.to_csv("submission_GRU.csv", index=False)


# ## Embedding Model

# In[19]:



EMBEDDING_FILE =open("glove.840B.300d.txt", encoding="utf8") 

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in EMBEDDING_FILE)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[20]:


inp = Input(shape=(max_words,))
x = Embedding(max_features, embed_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(2, activation = "softmax")(x)
model2 = Model(inputs = inp, outputs=x)
model2.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model2.summary())


# In[21]:


model2.fit(x_train, y_train, batch_size = 512, epochs=10, validation_data=(x_val, y_val))


# In[22]:


pred2=np.round(np.round(np.clip(model2.predict(x_val), 0, 1)))
print(f1_score(y_val, pred2, average=None))


# In[23]:


embedding_matrix.shape


# In[24]:


pred2=np.round(np.clip(model2.predict(x_test), 0, 1)).astype(int)
pred2 = pd.DataFrame(pred2)
pred2 = pred2.idxmax(axis=1)
submission_EMB = pd.DataFrame({'id':test['id'], 'label':pred2})
submission_EMB.to_csv("submission_EMB.csv", index=False)


# ## LSTM Model

# In[25]:


model3= Sequential()
model3.add(Embedding(max_features,embed_dim,weights=[embedding_matrix],mask_zero=True))
model3.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model3.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5, return_sequences=False))
model3.add(Dense(2,activation='softmax'))
model3.compile(loss='categorical_crossentropy',optimizer= Adam(lr=0.001), metrics=['accuracy'])
model3.summary()


# In[26]:


model3.fit(x_train, y_train, batch_size = 512, epochs=10, validation_data=(x_val, y_val))


# In[27]:


pred3=np.round(np.round(np.clip(model3.predict(x_val), 0, 1)))
print(f1_score(y_val, pred3, average=None))


# In[28]:


pred3=np.round(np.clip(model3.predict(x_test), 0, 1)).astype(int)
pred3 = pd.DataFrame(pred3)
pred3 = pred3.idxmax(axis=1)
submission_LSTM = pd.DataFrame({'id':test['id'], 'label':pred3})
submission_LSTM.to_csv("submission_LSTM.csv", index=False)


# ## CNN MOdel

# In[29]:


model4= Sequential()
model4.add(Embedding(max_features,embed_dim,weights=[embedding_matrix]))
model4.add(Dropout(0.2))
model4.add(Conv1D(64,kernel_size=3,padding='same', activation='relu', strides=1))
model4.add(GlobalMaxPooling1D())
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.2))
model4.add(Dense(2,activation='softmax'))
model4.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model4.summary()


# In[30]:


model4.fit(x_train,y_train, batch_size=512, epochs=20,validation_data=(x_val, y_val))


# In[31]:


pred4=np.round(np.clip(model4.predict(x_val), 0, 1))
print(f1_score(y_val, pred4, average=None))


# In[32]:


pred4=np.round(np.clip(model4.predict(x_test), 0, 1)).astype(int)
pred4=pd.DataFrame(pred4)
pred4=pred4.idxmax(axis=1)
submission_CNN = pd.DataFrame({'id':test['id'], 'label': pred4})
submission_CNN.to_csv("submission_CNN.csv", index= False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




