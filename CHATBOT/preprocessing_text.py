import nltk
import string
import json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def preprocessing(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
        
    with open('stopwords.json') as f:
        stopwords=json.load(f)
    
    with open('slangwords.json') as f:
        slangwords=json.load(f)
        
    punctuation = string.punctuation + '’'
    # lowercase
    text = text.lower()
    # hapus angka
    text = text.translate(str.maketrans(string.digits,' '*len(string.digits)))
    # hapus tanda baca
    text = text.translate(str.maketrans(punctuation,' '*len(punctuation)))
    # hapus whitespace
    text = text.strip(" ")
    
    # repair slangword
    slang = []
    formal = []
    
    for slangword in slangwords['slang'].items():
        slang.append(slangword[1])

    for formalword in slangwords['formal'].items():
        formal.append(formalword[1])
    text = text.split()
    for index,item in enumerate(text):
        if item in slang:
            slang_index = slang.index(item)
            text[index] = formal[slang_index]
    text = " ".join(text)
    
    # remove stopwords
    if type(text) == str:
        text = text.split()
    text = [word for word in text if word not in stopwords]
    text = " ".join(text)

    # stemming
    text = stemmer.stem(text)
        
    # tokenization
#     text = text.split()
    
    return text