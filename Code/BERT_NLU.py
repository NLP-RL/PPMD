import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
#tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()
from bert.tokenization import FullTokenizer
import numpy as np
from bert import tokenization as bert_tokenization
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Multiply, TimeDistributed, Dropout
import json
from itertools import chain
import argparse
import os
import pickle
from sklearn import metrics
from tensorflow.python.keras import backend as K
K.set_session

#-----Ashok
#Ch : tf downgrade
graph = tf.compat.v1.get_default_graph()
# from bottom copied and pasted above
sess = tf.compat.v1.Session()

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layer=12, bert_path='https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1', **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layer
        self.trainable = True
        self.output_size = 768
        self.bert_path = bert_path
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path, trainable=self.trainable, name="{}_module".format(self.name))
        trainable_vars = self.bert.variables
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers:]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        # Add non_trainable weights:
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs] #cast the variables to int32 tensor
        input_ids, input_mask, segment_ids, valid_positions = inputs
        bert_inputs = dict(input_ids=input_ids, ## we can use 'convert_tokens_to_ids' function to get the ids from tokens
                           input_mask=input_mask,
                           segment_ids=segment_ids)
        result = self.bert(inputs=bert_inputs, signature='tokens', as_dict=True)
        return result['pooled_output'], result['sequence_output']

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_size)

class NLUModel:

    def __init__(self):
        self.model = None

    def visualize_metric(self, history_dic, metric_name):
        plt.plot(history_dic[metric_name])
        legend = ['train']
        if 'val_' + metric_name in history_dic:
            plt.plot(history_dic['val_'+metric_name])
            legend.append('test')
        plt.title('model_' + metric_name)
        plt.ylabel(metric_name)
        plt.xlabel('epoch')
        plt.legend(legend, loc='upper left')
        plt.show()

    def predict(self, x):
        #graph = tf.get_default_graph()
        global graph
        global sess
        with graph.as_default():
            K.set_session(sess)
            return self.model.predict(x)
        #return self.model._make_predict_function(self,x)

    def save(self, model_path):
        self.model.save(model_path)

    def load(model_path, custom_objects=None):
        new_model = NLUModel()
        new_model.model = load_model(model_path, custom_objects=custom_objects)
        return new_model

    def predict_slots_intent(self, x, slots_tokenizer, intents_label_encoder):
        if len(x.shape) == 1:
            x = x[np.newaxis, ...]

        y1, y2 = self.predict(x)
        intents = np.array([intents_label_encoder.inverse_transform([np.argmax(y2[i])])[0] for i in range(y2.shape[0])])
        slots = []
        for i in range(y1.shape[0]):
            y = [np.argmax(i) for i in y1[i]]
            slot = []
            for i in y:
                if i == 0:
                  pass

                else:
                    slot.append(slots_tokenizer.index_word[i])
            slots.append(slot[:len(x[i])+1])
        return intents, slots


class TagsVectorizer:

    def __init__(self):
        pass

    def tokenize(self, tags_str_arr):
        return [s.split() for s in tags_str_arr]

    def fit(self, train_tags_str_arr, val_tags_str_arr):
        ## in order to avoid, in valid_dataset, there is tags which not exit in train_dataset. like: ATIS datset
        self.label_encoder = LabelEncoder()
        data = ["[padding]", "[CLS]", "[SEP]"] + [item for sublist in self.tokenize(train_tags_str_arr) for item in sublist]
        data = data + [item for sublist in self.tokenize(val_tags_str_arr) for item in sublist]
        ## # data:  ["[padding]", "[CLS]", "[SEP]", all of the real tags]; add the "[padding]", "[CLS]", "[SEP]" for the real tag list
        self.label_encoder.fit(data)

    def transform(self, tags_str_arr, valid_positions):
        ## if we set the maximum length is 50, then the seq_length is 50; otherwise, it will be equal to the maximal length of dataset
        seq_length = valid_positions.shape[1] # .shape[0]: number of rows, .shape[1]: number of columns
        data = self.tokenize(tags_str_arr)
        ## we added the 'CLS' and 'SEP' token as the first and last token for every sentence respectively
        data = [self.label_encoder.transform(["[CLS]"] + x + ["[SEP]"]).astype(np.int32) for x in data] #upper 'O', not 0
        data=np.array(data)
        #print(data)
        output = np.zeros((len(data), seq_length))
        #print(output)
        for i in range(len(data)):
            idx = 0
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                  if idx>=len(data[i]):
                    output[i][j]=0
                  else:
                    output[i][j] = data[i][idx]
                    idx += 1
        return output

    def inverse_transform(self, model_output_3d, valid_position):
        ## model_output_3d is the predicted slots output of trained model
        seq_length = valid_position.shape[1]
        slots = np.argmax(model_output_3d, axis=-1)
        slots = [self.label_encoder.inverse_transform(y) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(seq_length):
                if valid_position[i][j] == 1:
                  y.append(str(slots[i][j]))

            output.append(y)

        return output

    def load(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    if not os.path.exists('NLU_Weight/'):
        print('Folder "%s" not exist' % load_folder_path)
    with open(os.path.join('NLU_Weight/', 'tags_vectorizer.pkl'), 'rb') as handle:
        tags_vectorizer = pickle.load(handle)
        slots_num = len(tags_vectorizer.label_encoder.classes_)
    with open(os.path.join('NLU_Weight/', 'intents_label_encoder.pkl'), 'rb') as handle:
        intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)
    train_tags_str_arr = ['O O B-X B-Y', 'O B-Y O']
    val_tags_str_arr = ['O O B-X B-Y', 'O B-Y O XXX']
    valid_positions = np.array([[1, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1]])

    vectorizer = TagsVectorizer()
    vectorizer.fit(train_tags_str_arr, val_tags_str_arr)
    data = vectorizer.transform(train_tags_str_arr, valid_positions)
    #print(data, vectorizer.label_encoder.classes_)


BertTokenizer = bert_tokenization.FullTokenizer


class BERTVectorizer:

    def __init__(self, sess, bert_model_hub_path):
        self.sess = sess
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module()
        #print(bert_model_hub_path)

    def create_tokenizer_from_hub_module(self):
        # get the vocabulary and lowercasing or uppercase information directly from the BERT tf hub module
        bert_module = hub.Module(self.bert_model_hub_path)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        vocab_file, do_lower_case = self.sess.run(
            [
                tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"]
            ]
        )
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case) #do_lower_case=True

    def tokenize(self, text:str): ## tokenize every sentence
        words = text.split()
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)

        return tokens, valid_positions

    def transform(self, text_arr):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        for text in text_arr:
            ids, mask, seg_ids, valid_pos = self.__vectorize(text)
            input_ids.append(ids)
            input_mask.append(mask)
            segment_ids.append(seg_ids)
            valid_positions.append(valid_pos)

        sequence_length = np.array([len(i) for i in input_ids])

        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=50, truncating='post', padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, maxlen=50, truncating='post', padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, maxlen=50, truncating='post', padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, maxlen=50, truncating='post', padding='post')


        return input_ids, input_mask, segment_ids, valid_positions, sequence_length

    def __vectorize(self, text:str):
        tokens, valid_positions = self.tokenize(text)

        ## insert the first token "[CLS]"
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        ## insert the last token "[SEP]"
        tokens.append('[SEP]')
        valid_positions.append(1)

        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids) ## The mask has 1 for real tokens and 0 for padding tokens.

        return input_ids, input_mask, segment_ids, valid_positions

'''
build and compile our model using the BERT layer
'''
class JointBertModel(NLUModel):
    def __init__(self, slots_num, intents_num, sess, num_bert_fine_tune_layers=12):
        self.slots_num = slots_num
        self.intents_num = intents_num
        self.num_bert_fine_tune_layers = num_bert_fine_tune_layers

        self.model_params = {
            'slots_num': slots_num,
            'intents_num': intents_num,
            'num_bert_fine_tune_layers': num_bert_fine_tune_layers
        }

        self.build_model()
        self.compile_model()

        self.initialize_vars(sess)


    def build_model(self):

        in_id = Input(shape=(None,), name='input_ids')
        in_mask = Input(shape=(None,), name='input_masks')
        in_segment = Input(shape=(None,), name='segment_ids')
        in_valid_positions = Input(shape=(None, self.slots_num), name='valid_positions')
        bert_inputs = [in_id, in_mask, in_segment, in_valid_positions]

        # the output of trained Bert
        bert_pooled_output, bert_sequence_output = BertLayer(n_fine_tune_layer=self.num_bert_fine_tune_layers, name='BertLayer')(bert_inputs)

        # add the additional layer for intent classification and slot filling
        intents_drop = Dropout(rate=0.1)(bert_pooled_output)
        intents_fc = Dense(self.intents_num, activation='softmax', name='intent_classifier')(intents_drop)

        slots_drop = Dropout(rate=0.1)(bert_sequence_output)
        slots_output = TimeDistributed(Dense(self.slots_num, activation='softmax'))(slots_drop)
        slots_output = Multiply(name='slots_tagger')([slots_output, in_valid_positions])

        self.model = Model(inputs=bert_inputs, outputs=[slots_output, intents_fc])

    def compile_model(self):
        optimizer = tf.keras.optimizers.Adam(lr=5e-5)
        # if the targets are one-hot labels, using 'categorical_crossentropy'; while if targets are integers, using 'sparse_categorical_crossentropy'
        losses = {
            'slots_tagger': 'sparse_categorical_crossentropy',
            'intent_classifier': 'sparse_categorical_crossentropy'
        }
        ## loss_weights: to weight the loss contributions of different model outputs.
        loss_weights = {'slots_tagger': 3.0, 'intent_classifier': 1.0}
        metrics = {'intent_classifier': 'acc'}
        self.model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        #self.model.summary()

    def fit(self, X, Y, validation_data=None, epochs=5, batch_size=32):
        X = (X[0], X[1], X[2], self.prepare_valid_positions(X[3]))
        if validation_data is not None:
            X_val, Y_val = validation_data
            validation_data = ((X_val[0], X_val[1], X_val[2], self.prepare_valid_positions(X_val[3])), Y_val)

        history = self.model.fit(X, Y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)

        self.visualize_metric(history.history, 'slots_tagger_loss')
        self.visualize_metric(history.history, 'intent_classifier_loss')
        self.visualize_metric(history.history, 'loss')
        self.visualize_metric(history.history, 'intent_classifier_acc')


    def prepare_valid_positions(self, in_valid_positions):
        ## the input is 2-D in_valid_position
        in_valid_positions = np.expand_dims(in_valid_positions, axis=2) ## expand the shape of the array to axis=2
        ## 3-D in_valid_position
        in_valid_positions = np.tile(in_valid_positions, (1,1,self.slots_num)) ##
        return in_valid_positions

    def predict_slots_intent(self, x, slots_vectorizer, intent_vectorizer, remove_start_end=True):
        valid_positions = x[3]
        x = (x[0], x[1], x[2], self.prepare_valid_positions(valid_positions))

        y_slots, y_intent = self.predict(x)

        ### get the real slot-tags using 'inverse_transform' of slots-vectorizer
        slots = slots_vectorizer.inverse_transform(y_slots, valid_positions)
        if remove_start_end: ## remove the first '[CLS]' and the last '[SEP]' tokens.
            slots = np.array([x[1:-1] for x in slots])

        ### get the real intents using 'inverse-transform' of intents-vectorizer
        intents = np.array([intent_vectorizer.inverse_transform([np.argmax(y_intent[i])])[0] for i in range(y_intent.shape[0])])
        return slots, intents

    def initialize_vars(self, sess):
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())
        K.set_session(sess)

    def save(self, model_path):
        with open(os.path.join(model_path, 'params.json'), 'w') as json_file:
            json.dump(self.model_params, json_file)
        self.model.save_weights(os.path.join(model_path, 'joint_bert_model.h5'))

    def load(load_folder_path, sess):
        with open(os.path.join(load_folder_path, 'params.json'), 'r') as json_file:
            model_params = json.load(json_file)

        slots_num = model_params['slots_num']
        intents_num = model_params['intents_num']
        num_bert_fine_tune_layers = model_params['num_bert_fine_tune_layers']

        new_model = JointBertModel(slots_num, intents_num, sess, num_bert_fine_tune_layers)
        new_model.model.load_weights(os.path.join(load_folder_path, 'joint_bert_model.h5'))
        return new_model


parser = argparse.ArgumentParser('Evaluating the Joint BERT model')
parser.add_argument('--model', '-m', help='path to joint bert model', type=str, required=True)
parser.add_argument('--data', '-d', help='path to test data', type=str, required=True)
parser.add_argument('--batch', '-bs', help='batch size', type=int, default=128, required=False)
parser.add_argument('--pre_intents', '-pre_is', help='teh file name of saving predicted intents', type=str, required=True)
parser.add_argument('--pre_slots', '-pre_sls', help='the file name of saving predicted slots/tags', type=str, required=True)

args = parser.parse_args('-m model -d data -pre_is pre_intents -pre_sls pre_slots'.split())
load_folder_path = args.model
data_folder_path = args.data
batch_size =64
pre_intens_name = args.pre_intents
pre_slots_name = args.pre_slots

#sess = tf.compat.v1.Session()

bert_model_hub_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
bert_vectorizer = BERTVectorizer(sess, bert_model_hub_path)

## loading the model
#print('Loading models ....')
# if not os.path.exists('Weight/'):
#     print('Folder "%s" not exist' % load_folder_path)

# with open(os.path.join('Weight/', 'tags_vectorizer.pkl'), 'rb') as handle:
#     tags_vectorizer = pickle.load(handle)
#     slots_num = len(tags_vectorizer.label_encoder.classes_)
# with open(os.path.join('Weight/', 'intents_label_encoder.pkl'), 'rb') as handle:
#     intents_label_encoder = pickle.load(handle)
#     intents_num = len(intents_label_encoder.classes_)
#print(tags_vectorizer)
model2 = JointBertModel.load('NLU_Weight/', sess)

def predict2(utterance1):


    if not os.path.exists('NLU_Weight/'):
        print('Folder "%s" not exist' % load_folder_path)
    with open(os.path.join('NLU_Weight/', 'tags_vectorizer.pkl'), 'rb') as handle:
        tags_vectorizer = pickle.load(handle)
        slots_num = len(tags_vectorizer.label_encoder.classes_)
    with open(os.path.join('NLU_Weight/', 'intents_label_encoder.pkl'), 'rb') as handle:
        intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)
    train_tags_str_arr = ['O O B-X B-Y', 'O B-Y O']
    val_tags_str_arr = ['O O B-X B-Y', 'O B-Y O XXX']
    valid_positions = np.array([[1, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1]])

    vectorizer = TagsVectorizer()
    vectorizer.fit(train_tags_str_arr, val_tags_str_arr)
    data = vectorizer.transform(train_tags_str_arr, valid_positions)



    data=[]
    data.append(utterance1)
    data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = bert_vectorizer.transform(data)

    predicted_tags, predicted_intents = model2.predict_slots_intent(
            [data_input_ids, data_input_mask, data_segment_ids, data_valid_positions],
            tags_vectorizer, intents_label_encoder, remove_start_end=True
        )

    s={}
    s2={}
    print('Predicted Slot mapping with dialogue config')
    if predicted_intents[0]=='request':
        print('predicted tags',predicted_tags)
        print('uttrance 1',utterance1)
        #for i in range(len(utterance1.split(" "))):
        for i in range(0,len(utterance1.split(" "))):
            if(predicted_tags[0][i]!='O'):
                s1=predicted_tags[0][i].split('-')
                if(s1[1] =='RAM'):
                    s1[1]= 'Ram'
                if(s1[1]=='Internal_RAM'):
                    s1[1] = 'IRam'
                if(s1[1]=='SIM'):
                    s1[1] = 'Sim'
                if(s1[1] == 'Weight_g'):
                    s1[1] = 'Weight'
                if(s1[1] == 'Released_Yr'):
                    s1[1] = 'RY'
                if(s1[1] =='Released_Month'):
                    s1[1] = 'RM'
                if(s1[1] =='Cost'):
                    s1[1] = 'Price'
                s2[s1[1]]="UNK"
    else:
        print('NLU  utterance1: ',utterance1)
        print('NLU predicted tags:',predicted_tags)
        for i in range(0, len(utterance1.split(" "))):
            if(predicted_tags[0][i]!='O'):
                tag=predicted_tags[0][i].split('-')
                vv = utterance1.split(" ")[i]
                if(tag[1] =='RAM'):
                    tag[1]= 'Ram'
                if(tag[1]=='Internal_RAM' or tag[1] == 'I_RAM' or tag[1] =='IRAM' ):
                    tag[1] = 'IRam'
                if(tag[1]=='SIM'):
                    tag[1] = 'Sim'
                if(tag[1] == 'Weight_g'):
                    tag[1] = 'Weight'
                if(tag[1] == 'Released_Yr'):
                    tag[1] = 'RY'
                if(tag[1] =='Released_Month'):
                    tag[1] = 'RM'
                if(tag[1] =='Cost'):
                    tag[1] = 'Price'

                ###Intensifier Resolution
                if(vv=='good' and (tag[1]=='P_Camera' or tag[1]=='S_Camera')):
                    print('iii')
                    vv = 16
                if(vv=='best' and (tag[1]=='P_Camera' or tag[1]=='S_Camera')):
                    print('iii')
                    vv = 23

                if(tag[1]=='Battery'):
                    print('iii')
                    vv = float(vv)
                if(vv=='good' and (tag[1]=='Battery')):
                    print('iii')
                    vv = 3000.0




                s[tag[1]]=vv



    k = predicted_intents[0]
    if(predicted_intents[0]=='info' or predicted_intents[0] =='f_info' or predicted_intents[0] =='reject' or predicted_intents[0] =='confirm' ):
       k = 'inform'
    if(predicted_intents[0]=='Thanks'):
       k = 'thanks'
    result= {
      "intent":k,
      "inform_slots":s,
      "request_slots":s2
  }
    return predicted_intents,predicted_tags,result
