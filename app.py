#-------------------- Deployment Modules------------------------#
import flask
from flask import Flask, jsonify, request, render_template
import joblib
import jsonify
import json
#-------------------- Deployment Modules------------------------#

#-------------------- Data Modules-----------------------------#
import numpy as np
import pandas as pd
import re
import json
import random
import math
import time
import unicodedata 
#import csv
import itertools
import os
import codecs
#-------------------- Data Modules-----------------------------#
#import spacy
#spacy_english = spacy.load('en_core_web_sm')

#-------------------- NLP Modules------------------------------#

#-----------------Machine Learning Modules--------------------#
import torch
from torch.jit import script, trace
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#from __future__ import division
#from __future__ import print_function
#from __future__ import unicode_literals
#from __future__ import absolute_import
#-----------------Machine Learning Modules--------------------#

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods = ['POST'])
def chat():

    class Vocabulary:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.index2word = {}
            self.word2count = {}
            self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token : 'EOS'}
            self.num_words = 3
        
        def addWord(self, w):
            if w not in self.word2index:
                self.word2index[w] = self.num_words
                self.index2word[self.num_words] = w
                self.word2count[w] = 1
                self.num_words += 1
            else:
                self.word2count[w] += 1
        
        def addSentence(self, sent):
            for word in sent.split(' '):
                self.addWord(word)
        
        def trim(self, min_cnt):
            if self.trimmed:
                return
            self.trimmed = True
            words_to_keep = []
            for key, value in self.word2count.items():
                if value > min_cnt:
                    words_to_keep.append(key)
            print('Words to Keep: {}/{} = {:.2f}%'.format(len(words_to_keep),len(self.word2count),len(words_to_keep)/len(self.word2count)))
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token : 'EOS'}
            self.num_words = 3
            for w in words_to_keep:
                self.addWord(w)

            
    class EncoderRNN(nn.Module):
        def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
            super(EncoderRNN, self).__init__()
            self.n_layers = n_layers
            self.hidden_size = hidden_size
            self.embedding = embedding

            self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                              dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

        def forward(self, input_seq, input_lengths, hidden=None):
            embedded = self.embedding(input_seq)
            packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
            outputs, hidden = self.gru(packed, hidden)
            # Unpack padding
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
            # Sum bidirectional GRU outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
            # Return output and final hidden state
            return outputs, hidden

    class Attn(nn.Module):
        def __init__(self, hidden_size):
            super(Attn, self).__init__()
            self.hidden_size = hidden_size

        def dot_score(self, hidden, encoder_output):
            return torch.sum(hidden * encoder_output, dim=2)

        def forward(self, hidden, encoder_outputs):
            attn_energies = self.dot_score(hidden, encoder_outputs)
            attn_energies = attn_energies.t()
            return F.softmax(attn_energies, dim=1).unsqueeze(1)

    class DecoderRNN(nn.Module):
        def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
            super(DecoderRNN, self).__init__()

            self.hidden_size = hidden_size
            self.output_size = output_size
            self.n_layers = n_layers
            self.dropout = dropout

            self.embedding = embedding
            self.embedding_dropout = nn.Dropout(dropout)
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
            self.concat = nn.Linear(2 * hidden_size, hidden_size)
            self.out = nn.Linear(hidden_size, output_size)

            self.attn = Attn(hidden_size)

        def forward(self, input_step, last_hidden, encoder_outputs):
            embedded = self.embedding(input_step)
            embedded = self.embedding_dropout(embedded)
            rnn_output, hidden = self.gru(embedded, last_hidden)
            attn_weights = self.attn(rnn_output, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            rnn_output = rnn_output.squeeze(0)
            context = context.squeeze(1)
            concat_input = torch.cat((rnn_output, context), 1)
            concat_output = torch.tanh(self.concat(concat_input))
            output = self.out(concat_output)
            output = F.softmax(output, dim=1)
            return output, hidden

    class GreedySearchDecoder(nn.Module):
        def __init__(self, encoder, decoder):
            super(GreedySearchDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def forward(self, input_seq, input_length, max_length):
            encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
            decoder_hidden = encoder_hidden[:decoder.n_layers]
            #decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
            #all_tokens = torch.zeros([0], device=device, dtype=torch.long)
            #all_scores = torch.zeros([0], device=device)
            decoder_input = torch.ones(1, 1, dtype=torch.long) * SOS_token
            all_tokens = torch.zeros([0], dtype=torch.long)
            all_scores = torch.zeros([0])            
            for _ in range(max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                decoder_input = torch.unsqueeze(decoder_input, 0)
            return all_tokens, all_scores


    def unicodeToASCII(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')          
            
    def cleanString(s):
        s = unicodeToASCII(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s        


    def indexFromSentence(voc, sent):
        return [voc.word2index[w] for w in sent.split(' ')] + [EOS_token]


    def evaluate(encoder, decoder, searcher, voc, sentence, max_length=10):
        indices = [indexFromSentence(voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indices])
        input_batch = torch.LongTensor(indices).transpose(0, 1)
        input_batch = input_batch
        #lengths = lengths.to(device)
        tokens, scores = searcher(input_batch, lengths, max_length)
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    model_name = 'chatbot_model'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.15
    batch_size = 64
    corpus_name = 'movie_corpus'
    max_length = 10
    voc = Vocabulary(corpus_name)
    loadFilename = 'D:\\PracticeProjects\\Chatbot\\chatbotAPI\\chatbot_model\\movie_corpus\\2-2_500\\4000_checkpoint.tar'
    checkpoint = torch.load(loadFilename)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    embedding_sd = checkpoint['embedding']
    embedding = nn.Embedding(voc.num_words, hidden_size)    
    embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = DecoderRNN(embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)    
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)    
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)
    #request_json = request.get_json(force=True)
    #input_review = str(request_json["input"])
    input_review = str(request.form.get('chatbox'))    
    input_sentence = ''
    #while(1):
    if input_review == 'quit':return 'exit'
    
    try:
        input_sentence = cleanString(input_review)
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]            
        #response = json.dumps({'response':' '.join(output_words)})
        response = ' '.join(output_words)
        return render_template('index.html', response = response)
    except KeyError:
        #response = json.dumps({'response':"Error: Unknown Word"})
        return render_template('index.html', response ='Error: Unknown Word') 
  

if __name__ == '__main__':
    app.run(port=5000, debug=True)