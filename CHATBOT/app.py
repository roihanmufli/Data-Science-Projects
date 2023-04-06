#Imports
from flask import Flask, render_template, request, jsonify
import nltk
import datetime
import numpy as np
import random
import json
import pickle
import pandas as pd
from preprocessing_text import preprocessing

df_user_response = pd.read_excel('dataset.xlsx',sheet_name='Doni')

list_intents = df_user_response['intent'].unique().tolist()
dict_response = {}

for list_intent in list_intents:
    dict_response[list_intent] = df_user_response[df_user_response['intent'] == list_intent]['response'].values.tolist()
    
with open("labels.pickle",'rb') as f:
    labels = pickle.load(f)
with open("tfidf.pickle", 'rb') as f:
	tfidf_vectorizer = pickle.load(f)
with open("label_encoder.pickle", 'rb') as f:
	le = pickle.load(f)
with open("model_rf.pickle", 'rb') as f:
	model = pickle.load(f)



app = Flask(__name__,template_folder='template')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    if message:
        message = preprocessing(message)
        tf_idf = tfidf_vectorizer.transform([message])
        output = model.predict_proba(tf_idf.toarray())[0]
        output_index = np.argmax(output)
        
        if output[output_index] < 0.63:
            response = 'Maaf saya tidak mengerti apa maksud kamu'
        response_tag = np.unique(le.inverse_transform(labels))[output_index]
        response = random.choice(dict_response[response_tag])
    return str(response)
	
if __name__ == "__main__":
	app.run()