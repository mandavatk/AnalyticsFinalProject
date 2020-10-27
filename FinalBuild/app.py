# -*- coding: utf-8 -*-
"""
Created on 10/26/2020

@author: Chris Wagner
"""
import os
import pandas as pd

import json

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource, request
import joblib

from GloveVec import GloveVectorizer
from sklearn.model_selection import train_test_split

import spacy
import pandas as pd
import matplotlib.pyplot as mplt
import seaborn as sn
import matplotlib.pyplot as plt                     
import numpy as np           
import plotly.offline as plyo

import random
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from collections import defaultdict
import string

from nltk.corpus import stopwords
from sklearn.metrics import log_loss
from tqdm import tqdm

stop = stopwords.words('english')

port = int(os.getenv('PORT', '5000'))

app = Flask(__name__)
api = Api(app)


# argument parsing
#parser = reqparse.RequestParser()
#parser.add_argument('query')

#modeling class for glove vectorizer
import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin


#text cleaning functions for running the text tester
def  remove_html(df, text):
    df[text] = df[text].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    return df

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

nlp = spacy.load('en_core_web_md')
# Clean text before feeding it to model
punctuations = string.punctuation

# Define function to cleanup text by removing personal pronouns, stopwords, puncuation and reducing all characters to lowercase 
def cleanup_text(docs, logging=False):
    texts = []
    for doc in docs:
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        #remove stopwords and punctuations 
        tokens = [tok for tok in tokens if tok not in stop and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)

class SentimentAnalysis(Resource):
    def get(self):
        ets_model = joblib.load('svm_model.pkl')
                
        # get the query parameters
        intext = request.args.get('testtext')
        
        # build a data frame for sending the input to the model
        dfTest = pd.DataFrame()

        dfTest = dfTest.append({"Text":intext}, ignore_index=True)
        
        #clean the data
        dfTest = remove_html(dfTest,"Text")
        dfTest['Text']=dfTest['Text'].apply(lambda x: remove_emoji(x))

        dfTest['token'] = cleanup_text(dfTest['Text'],logging=False)

        scrubbedInputTest = dfTest["token"]
        
        # make a prediction.  If the model fails due to no text fields matching, return -1
        try:
            pred_uc = ets_model.predict(scrubbedInputTest)
            output = str(pred_uc[0])
        except:
            output = "-1"
        
        return output
 

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(SentimentAnalysis, '/sentanalysis')



if __name__ == '__main__':
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=port)