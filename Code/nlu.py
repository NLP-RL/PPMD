from BERT_NLU import TagsVectorizer
from BERT_NLU import predict2
import pickle
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer

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




print('Enter a sentence:')
inp = input()
i , t ,a = predict2(inp)
print('User Action:',a)
s = load_predict(inp)
print(m-1)

