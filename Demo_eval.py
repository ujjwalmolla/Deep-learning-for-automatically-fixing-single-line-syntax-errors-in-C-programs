#import the libraries
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from operator import itemgetter
from keras.models import Model
from keras.layers import Input, LSTM, Dense

class Vocabulary:
  def __init__(self, name):
    self.name = name
    self.token2index = {"PAD":0 , "SOS":1 , "EOS":2}
    self.token2count = {}
    self.index2token = {0: "PAD", 1: "SOS", 2: "EOS"}
    self.num_tokens = 4
    self.num_token_line=0
    self.source_line=[]
    self.target_line=[]
    self.max_source_line_len=0
    self.max_target_line_len=0

  def add_token(self, token):
    if token not in self.token2index:
      self.token2index[token] = self.num_tokens
      self.token2count[token] = 1
      self.index2token[self.num_tokens] = token
      self.num_tokens += 1
    else:
      self.token2count[token] += 1
            
  def build_vocabulary(self, sourceLineTokens,targetLineTokens):        
      np.array(sourceLineTokens).reshape(-1,1)
      np.array(targetLineTokens).reshape(-1,1)

      for i in sourceLineTokens:
        self.source_line.append(i)
        s=eval(i) 
        if(self.max_source_line_len < len(s)):
          self.max_source_line_len=len(s)         
        for j in s:
          self.add_token(j)
      for i in targetLineTokens:
        self.target_line.append(i)
        s=eval(i)
        if(self.max_target_line_len < len(s)):
          self.max_target_line_len=len(s)           
        for j in s:
          self.add_token(j)
      
      return  self.token2index, self.token2count, self.index2token, self.num_tokens, self.source_line, self.target_line

  def get_top_k_token(self):
    top_k_token_count=dict(sorted(self.token2count.items(), key = itemgetter(1), reverse = True)[:top_k])
    top_k_token2index = {"PAD":0, "SOS":1, "EOS":2, "OOV_TOKEN":3}
    top_k_index2token = {0: "PAD", 1: "SOS", 2: "EOS", 3: "OOV_TOKEN"}
    num_count = 4
    for i in top_k_token_count:
      top_k_token2index[i] = num_count
      top_k_index2token[num_count] = i
      num_count += 1

    return top_k_token_count, top_k_token2index, top_k_index2token

#import the data sheet
data=pd.read_csv('train.csv')
top_k=1000
latent_dim=512
batch_size=64
epochs = 60
maxlen=30
file_in=sys.argv[1]
file_out=sys.argv[2]

#Build the vocabulary
vocab=Vocabulary("vocabulary")
token2index,token2count,index2token,num_tokens,source_line,target_line=vocab.build_vocabulary(data['sourceLineTokens'] , data['targetLineTokens'] )

token2count,token2index,index2token=vocab.get_top_k_token()
print("Vocabulary Created")

def convert_to_index_seq(token_line):
  token_index_seq=[]
  for i in token_line:
    tokens=eval(i)
    temp_list=[]     
    for tok in tokens:      
      if tok in token2index.keys():
        temp_list.append(token2index[tok])
      else:
        temp_list.append(3)
    token_index_seq.append(np.array([1]+temp_list+[2]))
  return np.array(token_index_seq)

"""**Convert to index Sequence (Training Set)**"""

token_index_sourceline=convert_to_index_seq(source_line)
token_index_targetline=convert_to_index_seq(target_line)

"""**PAD The Token index sequence (Trainning Set)**"""

#pad the token_index_seq (Training Set)
padded_source_line = tf.keras.preprocessing.sequence.pad_sequences(token_index_sourceline,padding='post', maxlen = maxlen)
padded_target_line = tf.keras.preprocessing.sequence.pad_sequences(token_index_targetline,padding='post', maxlen = maxlen)

"""**ONE HOT REPRESENTATION**"""

import numpy as np
encoder_input_data = np.zeros(
    (len(padded_source_line), maxlen, top_k+4), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(padded_source_line), maxlen, top_k+4), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(padded_source_line), maxlen, top_k+4), dtype="float32"
)
print(encoder_input_data.shape)
print(decoder_input_data.shape)
print(decoder_target_data.shape)


for i, (source_l, target_l) in enumerate(zip(padded_source_line, padded_target_line)):
    for t in range(len(source_l)):
      encoder_input_data[i, t, source_l[t]] = 1.0

    for t in range(len(target_l)):
      decoder_input_data[i, t, target_l[t]] = 1.0
      if t>0:
        decoder_target_data[i, t-1 , target_l[t]] = 1.0

"""**Convert to and PAD the index Sequence and One Hot Representation (Valid Set)**"""

import numpy as np
data_valid=pd.read_csv(sys.argv[1])
source_line_valid=data_valid['sourceLineTokens']
target_line_valid=data_valid['targetLineTokens']

#Convert to index sequence
token_index_sourceline_valid=convert_to_index_seq(source_line_valid)
token_index_targetline_valid=convert_to_index_seq(target_line_valid)

#pad the token_index_seq (Valid Set)
padded_source_line_valid = tf.keras.preprocessing.sequence.pad_sequences(token_index_sourceline_valid,padding='post', maxlen = maxlen)
padded_target_line_valid = tf.keras.preprocessing.sequence.pad_sequences(token_index_targetline_valid,padding='post', maxlen = maxlen)

#One Hot representation
encoder_input_data_valid = np.zeros(
    (len(padded_source_line_valid), maxlen, top_k+4), dtype="float32"
)
decoder_input_data_valid = np.zeros(
    (len(padded_source_line_valid), maxlen, top_k+4), dtype="float32"
)
decoder_target_data_valid = np.zeros(
    (len(padded_source_line_valid), maxlen, top_k+4), dtype="float32"
)


for i, (source_l, target_l) in enumerate(zip(padded_source_line_valid, padded_target_line_valid)):
    for t in range(len(source_l)):
      encoder_input_data_valid[i, t, source_l[t]] = 1.0

    for t in range(len(target_l)):
      decoder_input_data_valid[i, t, target_l[t]] = 1.0
      if t>0:
        decoder_target_data_valid[i, t-1 , target_l[t]] = 1.0


"""**Load the Model**"""

from tensorflow import keras
model = keras.models.load_model("myModel.h5")

# load weights
#model.load_weights("weights.best.hdf5")

encoder_inputs = model.input[0]
encoder_outputs, state_y_enc, state_x_enc = model.layers[2].output
encoder_states = [state_y_enc, state_x_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]
decoder_state_input_y = Input(shape=(latent_dim,), name="input_9")
decoder_state_input_x = Input(shape=(latent_dim,), name="input_10")
decoder_states_inputs = [decoder_state_input_y, decoder_state_input_x]

decoder_lstm = model.layers[3]
decoder_outputs, state_y_dec, state_x_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_y_dec, state_x_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

"""**Model Evaluation**"""

model.evaluate([encoder_input_data_valid, decoder_input_data_valid],
    decoder_target_data_valid,verbose=2)

#-----------------Function for predicting and decoding output------------------------------- 
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, top_k+4))
    target_seq[0, 0, 1] = 1.0
    stop_condition = False
    decoded_Token = []
    while not stop_condition:
        output_tokens, y, x = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_Token = index2token[sampled_token_index]      
        if sampled_Token == 'EOS' or len(decoded_Token) > maxlen:
          stop_condition = True
        else:
          if sampled_Token not in list(['SOS','EOS','PAD']):
            decoded_Token.append(sampled_Token)  
        target_seq = np.zeros((1, 1, top_k+4))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [y, x]
    return decoded_Token


#--------------Token Decoding for valid data Set------------------
Fixed_Token=[]
print("Predicting the tokens")
for seq_index in range(len(source_line_valid)):
    # Take one sequence for decoding (part of the valid set)
    input_seq = encoder_input_data_valid[seq_index : seq_index + 1]
    decoded_Token = decode_sequence(input_seq)
    Fixed_Token.append(decoded_Token)
    
    #print("-------------------------------------------")
    print(seq_index)
    #print("Input Token  :", source_line_valid[seq_index])
    #print("Target Token :", target_line_valid[seq_index])
    #print("Decoded Token:", decoded_Token)

print("Predicted Tokens are:")
print(Fixed_Token)
#------------Create the CSV outfile containing FixedToken-------------#
import csv
import pandas as pd

with open(file_in, 'r') as read_f, \
        open(file_out, 'w', newline='') as write_f:
        csv_reader = csv.reader(read_f)
        csv_writer = csv.writer(write_f)
        i = 0
        for row in csv_reader:
          if(i==0):
            row.append("FixedToken")
          else:
            row.append(Fixed_Token[i-1])          
          csv_writer.writerow(row)
          i += 1
          if(i>len(Fixed_Token)):
            break
print("Output File created ")
