import streamlit as st
from duckduckgo_search import DDGS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import pickle
import re
import json

def predict_code(desc):
    # retrieve relevant chunks
    retrieved_docs = db.similarity_search(desc, k=4)
    context = r''
    for doc in retrieved_docs:
        context += ' ' + doc.page_content.replace("'",'')
        chunks = context
        context = re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', context)
        context = ''.join((i for i in context if 0x20 <= ord(i) < 127))
    # # LLM Predict
    output = DDGS().chat(f'''Provide the numerical code (4 digit numbers) for this product description:
            {desc}, using only the context provided found 
            between the tags [info] and [/info]. The numerical codes are found in the context within parenthesis, following the product categories.
            [info]
            {context}
            [/info]
            For the numerical code, return ONLY one 4 digit number.
            Output strictly in JSON format with keys: numerical_code, LLM Reasoning. 
            ''', model='gpt-3.5')
    output = json.loads(re.sub(r'(?<!\\)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'', output))
    return [output['numerical_code'], output['LLM Reasoning']], chunks

# Header
st.markdown('# LLM HTS Prediction (Chp 61)')

# Get input
desc = st.text_input('Enter product decription: ')

# load vector store
@st.cache_data
def load_vector_store():
    with open(r"chp61_hts_info_store", 'rb') as f:
        pkl = pickle.load(f)
    embeddings = FastEmbedEmbeddings()
    db = FAISS.deserialize_from_bytes(
        embeddings=embeddings, serialized=pkl
    )
    return db 

db = load_vector_store()

if st.button('Submit'):
    res_lst, chunks = predict_code(desc)
    st.markdown(f'Retrieved chunks: {chunks}')
    st.markdown(f"Predicted code: {'61' + str(res_lst[0])}")
    st.write(f'LLM reasoning: {res_lst[1]}')
else:
    st.write('Click button to see price prediction')