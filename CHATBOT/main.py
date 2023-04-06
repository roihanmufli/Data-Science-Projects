import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.utils import compute_class_weight
from tensorflow.keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing_text import preprocessing
import pickle
import random
import warnings
warnings.filterwarnings("ignore")


df_train = pd.read_excel('dataset.xlsx',sheet_name='main')
df_test = pd.read_excel('dataset.xlsx',sheet_name='test')


df_train['preprocessed'] = df_train['text'].apply(lambda x:preprocessing(x))
df_test['preprocessed'] = df_test['text'].apply(lambda x:preprocessing(x))

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer = tfidf_vectorizer.fit(df_train['preprocessed'])

# save tfidf vectorizer
with open('tfidf.pickle', 'wb') as tfidf:
    pickle.dump(tfidf_vectorizer, tfidf)

doc_vec_train = tfidf_vectorizer.transform(df_train['preprocessed'])
doc_vec_test = tfidf_vectorizer.transform(df_test['preprocessed'])

df2_train = pd.DataFrame(doc_vec_train.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names_out())

df2_test = pd.DataFrame(doc_vec_test.toarray().transpose(),
                   index=tfidf_vectorizer.get_feature_names_out())

le = LabelEncoder()
le = le.fit(df_train['intent'])

# save label encoder
with open('label_encoder.pickle', 'wb') as label_encoder:
    pickle.dump(le, label_encoder)
    
y_train = le.transform(df_train['intent'])

# save label 
with open('labels.pickle', 'wb') as labels:
    pickle.dump(y_train, labels)
    
y_test = le.transform(df_test['intent'])

X_train = df2_train.transpose()
X_test = df2_test.transpose()

# training
modelrf = RandomForestClassifier(class_weight='balanced',random_state=10)
modelrf.fit(X_train, y_train)

## save model
with open('model_rf.pickle', 'wb') as model:
    pickle.dump(modelrf, model)
    
pred = modelrf.predict(X_test)
print(classification_report(y_test, pred, target_names=np.unique(le.inverse_transform(y_train))))