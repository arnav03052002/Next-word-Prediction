import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Loading LSTM Model

model=load_model('LSTM_next_word.h5')

# Loading the tokenizer

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def predict_next_word(model,tokenizer,text,max_seq_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list], maxlen=max_sequence_len-1,padding='pre')
    predict=model.predict(token_list,verbose=0)
    predicted_next_word=np.argmax(predict,axis=1)
    for word, index in tokenizer.word_index.items():
        if index==predicted_next_word:
            return word
    return None

# Streamlit app

st.title("Next word prediction")
input_text=st.text_input("Enter the sequence of Words","The tree is")
if st.button('Predict The next word'):
    max_sequence_len=model.input_shape[1]+1
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word Prediction:{next_word}")
