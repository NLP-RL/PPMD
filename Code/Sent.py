import pandas as pd
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Bidirectional,Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
import re
import matplotlib.pyplot as plt
from keras.models import model_from_json
from sklearn.utils import class_weight
import os
import tensorflow as tf
#from bert import tokenization as bert_tokenization
import pickle
#from BERT_NLU import TagsVectorizer
#from BERT_NLU import predict2


###Data Loading and PreProcessing
data = pd.read_csv('SentData/final.csv')
sent = pd.read_csv('SentData/sent.csv')
utt = []
senti = []
for i in range(0,len(sent)):
    if(sent['User'][i]!=sent['Sentiment'][i]):
        utt = utt + [sent['User'][i]]
        if(sent['Sentiment'][i]=='nan'):
            senti  = senti + ['Neu']
        elif(sent['Sentiment'][i]!='Pos' and sent['Sentiment'][i]!='Neu' and sent['Sentiment'][i]!='Neg'):
             senti  = senti + ['Neu']
        else:
            senti  = senti + [sent['Sentiment'][i]]
data['Utterance'] = data['Utterance'].apply(lambda x: x.lower())
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(utt)
X = tokenizer.texts_to_sequences(utt)
X = pad_sequences(X,maxlen = 40)
Y_sent = pd.get_dummies(senti)
X_train, X_test, Y_train_sent, Y_test_sent = train_test_split(X,Y_sent, test_size = 0.20, random_state = 36)
embed_dim = 256
lstm_out = 196
sent_model = Sequential()
sent_model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
sent_model.add(SpatialDropout1D(0.4))
sent_model.add(Bidirectional(LSTM(lstm_out)))
sent_model.add(Dropout(0.3))
sent_model.add(Dense(3,activation='softmax'))
sent_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

def load_predict(s):
	with open('SentData/senttokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
	json_file = open('SentData/sent_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	sent_loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	sent_loaded_model.load_weights("SentData/sent_model.h5")
	ss = ['Negative','Neutral','Positive']
	utt = [s]
	#vectorizing the tweet by the pre-fitted tokenizer instance
	utt = tokenizer.texts_to_sequences(utt)
	#padding the tweet to have exactly the same shape as `embedding_2` input
	utt = pad_sequences(utt, maxlen=40, dtype='int32', value=0)
	#print(utt)
	sentiment = sent_loaded_model.predict(utt,batch_size=1,verbose = 2)[0]
	k = list(sentiment)
	m = np.argmax(k)
	return m-1




def train_save_test():
	class_weights = class_weight.compute_class_weight('balanced',np.unique(senti),senti )
	batch_size = 64
	history_sent = sent_model.fit(X_train,Y_train_sent,class_weight=class_weights, epochs = 15, batch_size=batch_size, verbose = 1)
	# plt.plot(history_sent.history['accuracy'])
	# plt.title('Sentiment Classification')
	# plt.ylabel('Accuracy')
	# plt.xlabel('Epoch')
	# plt.show()
	Y_sent_pred = sent_model.predict_classes(X_test,batch_size = batch_size)
	df_test = pd.DataFrame({'true': Y_test_sent.values.tolist(), 'pred':Y_sent_pred})
	df_test['true'] = df_test['true'].apply(lambda x: np.argmax(x))
	print("confusion matrix",confusion_matrix(df_test.true, df_test.pred))
	print(classification_report(df_test.true, df_test.pred))
	sent_model_json = sent_model.to_json()
	with open("SentData/sent_model.json", "w") as json_file:
	    json_file.write(sent_model_json)
	# serialize weights to HDF5
	sent_model.save_weights("SentData/sent_model.h5")

if __name__ == "__main__":
	#train_save_test()
	print('Enter a Sentence: ')
	utt = input()
	k = load_predict(utt)
	print(k)


