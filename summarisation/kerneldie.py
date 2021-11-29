# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

from os import listdir
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.tokenize import word_tokenize


from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
unprocessed_file_name = 'unprocessed_file.pkl'
cleaned_file_name ='cleaned_file.pkl'
cnn = "/Users/rohankilledar/Documents/MSc Artificial Intelligence/repos/summarisation/cnn/stories"
path_of_downloaded_files = "/Users/rohankilledar/Documents/MSc Artificial Intelligence/repos/Natural Language Processing/glove.6B.50d.txt"
filename = "glove.6B.50d.txt"
embedding_dim = 50
data_size = 2000
hidden_size = 50


# %%
glove_file = datapath(path_of_downloaded_files)
word2vec_glove_file = get_tmpfile(filename)
glove2word2vec(glove_file, word2vec_glove_file)
word_vectors = KeyedVectors.load_word2vec_format(word2vec_glove_file)
print('loaded %s word vectors from %s.' % (len(word_vectors.key_to_index),filename ))


# %%
SOS_token = 0
EOS_token = 1

sos = 'sos'
eos = 'eos'



class Lang:
    def __init__(self, name):
        self.name = name
        self.words = []
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0  # Count SOS and EOS
        self.glove = {}
        self.embedding_dim = word_vectors.get_vector('office').shape[0]

    def addSentence(self, sentence):
        for word in word_tokenize(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.words.append(word)
            if word not in self.glove:
                self.glove[word] = np.random.normal(scale=0.6 , size= (self.embedding_dim,))
        else:
            self.word2count[word] += 1

    def addPretrained(self,path_of_downloaded_files):
        with open(path_of_downloaded_files) as f:
            for indx,l in enumerate(f):
                line = l.split()
                word = line[0]
                if word == sos:
                    first_word = self.words[0]
                    self.words.append(first_word)
                    self.words[0] = sos
                    self.word2index[first_word] = indx
                    self.word2index[sos] = 0
                    self.index2word[indx] = first_word
                    self.index2word[0] = sos
                    self.glove[sos] = word_vectors[indx]
                    self.glove[first_word] = word_vectors[0]
                    self.word2count[sos] = 1
                    self.n_words += 1

                elif word == eos:
                    sec_word = self.words[1]
                    self.words.append(sec_word)
                    self.words[1] = eos
                    self.index2word[indx] = sec_word
                    self.index2word[1] = eos
                    self.word2index[sec_word] = indx
                    self.word2index[eos] = 1
                    self.glove[eos] = word_vectors[indx]
                    self.glove[sec_word] = word_vectors[1]
                    self.word2count[eos] = 1
                    self.n_words += 1
                else:
                    self.words.append(word)
                    self.word2count[word] = 1
                    self.word2index[word] = indx
                    self.index2word[indx]=word
                    self.glove[word] = word_vectors[indx]
                    self.n_words += 1
                    


# %%
def decontracted(phrase): 
    phrase = re.sub(r"won't", "will not", phrase) 
    phrase = re.sub(r"can\'t", "can not", phrase)  
    phrase = re.sub(r"n\'t", " not", phrase)  
    phrase = re.sub(r"\'re", " are", phrase)  
    phrase = re.sub(r"\'s", " is", phrase)  
    phrase = re.sub(r"\'d", " would", phrase)  
    phrase = re.sub(r"\'ll", " will", phrase)  
    phrase = re.sub(r"\'t", " not", phrase)  
    phrase = re.sub(r"\'ve", " have", phrase)  
    phrase = re.sub(r"\'m", " am", phrase)  
    return phrase


# %%
# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    
    
    dash_indx = s.find('(CNN) --')
    if dash_indx>=0: #and dash_indx<=20:
        s = s[dash_indx+len('(CNN) --'):]
        
    s=decontracted(s)
    s = unicodeToAscii(s.lower().strip())

    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# %%
def read_file(filename):
    file = open(filename, encoding= 'UTF-8')
    text = file.read()
    file.close()
    return text

def split_text(article):
    indx = article.index('@highlight')
    story = article[:indx]
    highlight = article[indx:].split('@highlight')

    highlight = ". ".join([h.strip() for h in highlight if len(h)>0])
    return story,highlight


# %%
def read_all(folder):
    dataset = list()

    for file in tqdm(listdir(folder)):
        filename = folder + '/' + file
        article = read_file(filename)
        story,highlight = split_text(article)

        dataset.append({'story':story, 'highlight':highlight})
    
    return dataset

dataset = read_all(cnn)


# %%
#saving dataset for cleaning
import pickle
output_file = open(unprocessed_file_name,'wb')
pickle.dump(dataset, output_file)
output_file.close()


# %%

dataset = pd.read_pickle(unprocessed_file_name)
#reducing the dataset for initial testing of model
dataset = dataset[:data_size]


# %%
df = pd.DataFrame(dataset)


# %%
df['story'] = df['story'].apply(lambda x: normalizeString(x))
df['highlight'] = df['highlight'].apply(lambda x: normalizeString(x))


# %%
df['highlight'] = df['highlight'].apply(lambda x: sos + " " + x )


# %%
df['word_count_text'] = df['story'].apply(lambda x: len(str(x).split()))
df['highlight_count'] = df['highlight'].apply(lambda x: len(str(x).split()))
from math import floor
print("The mean word count length of text article is " + str(floor(df['word_count_text'].mean())))
print("The mean word count length of summary/highlight is " + str(floor(df['highlight_count'].mean())))

max_article_len = floor(df['word_count_text'].max())
max_summary_len = floor(df['highlight_count'].max())
print("The max word count length of text article is " + str(max_article_len))
print("The max word count length of summary/highlight is " + str(max_summary_len))


# %%
df.drop('word_count_text', axis=1, inplace=True)
df.drop('highlight_count', axis=1, inplace=True)


# %%
with open("processed_data.pkl","wb") as save_path:
    pickle.dump(df, save_path)


# %%
vocab = Lang('vocab')
vocab.addPretrained(path_of_downloaded_files)
def prepareData(vocab):
    for indx,row in tqdm(df.iterrows()):
        vocab.addSentence(row['story'])
        vocab.addSentence(row['highlight'])
    print("counted Words:")
    print(vocab.name, vocab.n_words)
prepareData(vocab)


# %%
with open("vocab.pkl","wb") as save_path:
    pickle.dump(df, save_path)


# %%
pairs = df.values.tolist()


# %%
def indexesFromSentence(lang, sentence):
    # return [lang.word2index[word] for word in sentence.split(' ')]
    return [lang.word2index[word] for word in word_tokenize(sentence)]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(vocab, pair[0])
    target_tensor = tensorFromSentence(vocab, pair[1])
    return (input_tensor, target_tensor)


# %%

def generate_weights(unique_word_corpus,glove, tens=True):
    matrix_len = len(unique_word_corpus)
    unique_word2indx = {}
    
    weight_matrix  = np.zeros((matrix_len, embedding_dim))
   
    words_found = 0

    for indx, word in enumerate(unique_word_corpus):
        try:
            unique_word2indx[word] = indx
            weight_matrix[indx] = glove[word]
            words_found +=1
        except KeyError:
            weight_matrix[indx] = np.random.normal(scale=0.6 , size= (embedding_dim,))
            unique_word2indx[word] = indx
    if tens:
        return torch.from_numpy(weight_matrix) , unique_word2indx
    else:
        return weight_matrix,unique_word2indx

weight_matrix, meh= generate_weights(vocab.words, vocab.glove)


# %%
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = 1

       # self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings = vocab.n_words,embedding_dim= embedding_dim)
        self.embedding = self.embedding.from_pretrained(weight_matrix)
        #self.embedding.from_pretrained(glove)
        self.lstm = nn.LSTM(input_size = hidden_dim, hidden_size = hidden_dim) #, num_layers = 1, bidirectional = True, batch_first = True)
        #self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):

        embedded = self.embedding(input).view(1,1,-1)
        output = embedded.float()
       # print(output)
        #print(hidden)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        #in case of LSTM we need h0 and c0 hence init this as a tuple (h0,c0) and passed as hidden to lstm and in case of gru its just h0
       return (torch.zeros(self.num_layer, 1, self.hidden_size, dtype=torch.float, device=device), torch.zeros(self.num_layer,1,self.hidden_size,dtype = torch.float, device= device))


# %%
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings= output_size,embedding_dim= hidden_size)
        self.embedding.from_pretrained(weight_matrix)
        self.lstm = nn.LSTM(input_size= hidden_size, hidden_size= hidden_size) #, num_layers = 1, bidirectional = False, batch_first = True)
        #self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(in_features= hidden_size , out_features=vocab.n_words)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1,1,-1)
        output = F.relu(embedded)
        
        #hidden is [2,1,50]
        #emb is [1,1,50]
        output, hidden = self.lstm(embedded, hidden)
        #output = self.softmax(self.out(output[0]))
        output = self.fc(output[0])
        
        output = self.softmax(output)
        return output, hidden


# %%
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_article_len):
        super(AttnDecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding.from_pretrained(weight_matrix)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
       
    
        AttnDecoderRNN.embedded = embedded
        
        AttnDecoderRNN.hidden = hidden
        AttnDecoderRNN.encoder_outputs = encoder_outputs
        
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0].view(1,-1)), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

                                 

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
       return (torch.zeros(1, 1, self.hidden_size, dtype=torch.int64, device=device), torch.zeros(1,1,self.hidden_size,dtype = torch.int64, device= device))


# %%
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# %%
teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attn, max_length=vocab.n_words):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attn :
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden= decoder(
                    decoder_input, decoder_hidden)
                #tar_tens = weight_matrix[target_tensor[di].item()].view(-1).float()
            loss += criterion(decoder_output, target_tensor[di])
            
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attn :
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden= decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            tar_tens = weight_matrix[target_tensor[di].item()].view(-1).float()
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break
    

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# %%
import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# %%
def trainIters(encoder, decoder, n_iters, attn=False, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    if attn:
        print("training with attention.")
    print(start)

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print('encoder optimizer init')
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    print('decoder optimizer init')
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    print('choosing training pairs:' + str(n_iters))
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    
    print('init criterion')
    criterion = nn.NLLLoss()
    # criterion = nn.MSELoss()
    
    print('starting iterations')
    
    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, attn)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# %%
def evaluate(encoder, decoder, sentence ,max_length=max_summary_len, attn=False): # please check what should be max_length
    with torch.no_grad():
        input_tensor = tensorFromSentence(vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(input_length, encoder.hidden_size, device=device)
        print('evaluation init')
        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            
            if attn:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden) 
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == EOS_token:
                decoded_words.append('sos')
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# %%
def evaluateRandomly(encoder, decoder, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        #print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0],attn=False)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# %%
hidden_size = 50
voc_size = len(vocab.words)
hidden_dim = 50
max_summary_size = 60
encoder1 = EncoderRNN(vocab.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, vocab.n_words).to(device)
#attn_decoder1 = AttnDecoderRNN(hidden_size, vocab.n_words, dropout_p=0.1).to(device)


# %%
trainIters(encoder1, decoder1, 5000, print_every=100)

# trainIters(encoder1, attn_decoder1,1000,True, print_every=10)



# %%
import pickle

# save model and other necessary modules
all_info_want_to_save = {
    'encoder': encoder1,
    'decoder': decoder1,
    'vocab': vocab
}

with open("train_model.pkl","wb") as save_path:
    pickle.dump(all_info_want_to_save, save_path)



# %%
import pickle

# load the saved file
with open('train_model.pkl','rb') as ff:
    saved_info = pickle.load(ff)
    
# extract the information from the saved file
encoder1 = saved_info['encoder']
decoder1 = saved_info['decoder']


# %%
evaluateRandomly(encoder1, decoder1)


