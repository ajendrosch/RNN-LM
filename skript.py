import numpy as np
from collections import Counter
import nltk
from nltk import word_tokenize
import nltk.data
import itertools
import random

import operator
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

#######################################################################

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '2000')) #8000
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '40'))#80
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '50'))#100
_MODEL_FILE = os.environ.get('MODEL_FILE')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

########################################################################

test_list = ['<s>','all', 'this', 'happened', 'more', 'or', 'less','</s>', '<s>', 'it', 'happened', 'too', 'quick', '</s>']

def getWords(doc):
	ret = list(set(doc))
	ret.sort
	# fuer jedes wort pruefe ob schon in liste ansonsten speicher in liste
	# Token fuer Satzanfang und ende, also vor und nach punkt
	return ret

def hotVec(word, wordlist):
	# pruefe stelle von word in wordlist und erzeuge One Hot Vector
	pos = wordlist.index(word)
	vec = np.zeros(len(wordlist))
	vec[pos] = 1
	return vec

def createBigram(input_list):
	# wenn mehr als zwei woerter in input_list
	# save erstes und zweites
	# delete erstes aus doc
	# return erstes und zweites
	return zip(input_list, input_list[1:])

def trainRNNLM(rnn, doc):
	bigram = createBigram(doc)
	while(len(tmp) > 0):
		data = bigram.pop(0)
		rnn.train(data[0], data[1])
	# data[0] = input
	# data[1] = output

def vectorToWord(vec, wordlist):
	# vector ist ausgabe von RNN nach softmax
	#suche maximum position in vec
	pos = vec.tolist().index(np.amax(vec))
	return wordlist[pos]

def randWord(vec, wordlist):
	return np.random.choice(wordlist, 1, p=vec)[0]

def createSentence(lm, bigram, wordlist, length):
	#beginne mit starttoken
	out = ['SENTENCE_START']
	for i in range(length):
		out.append(randWord(lm(out[i], bigram, wordlist),wordlist))
		# out.append(rand_bi_word(bigram))
	return out

def createSentence2(lm, bigram, wordlist):
	#beginne mit starttoken
	out = ['SENTENCE_START']
	while not (out[-1] == 'SENTENCE_END'):
		out.append(randWord(lm(out[-1], bigram, wordlist),wordlist))
		# out.append(rand_bi_word(bigram))
	return out

def bilm(word, bigram, wordlist):
	ret = np.zeros(len(wordlist))
	relevantpart = [elem[1] for elem in bigram if elem[0]==word]
	size = len(relevantpart)
	if (size > 0):
		anz = Counter(relevantpart)
		for i in anz.items():
			ret += hotVec(i[0], wordlist)* i[1]
		return ret/size
	else: (ret + 1)/len(ret)

def rand_bi_word(bigram):
	# waehlt zufaellig wort aus allen relevanten konstellationen aus
	relevantpart = [elem[1] for elem in bigram if elem[0]==word]
	return relevantpart[random.randint(0,len(relevantpart))]

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

########################################################################

#open document doc

seminar = open('Seminar_Text.txt', 'r')
sem = seminar.read().decode('utf-8').lower()
seminar.close()
## seminar.readline()

#tokens = word_tokenize(sem)

# jedes Token ein Satz
sentences = tokenizer.tokenize(sem)
#text = nltk.Text(sentences)

#vocabulary_size = 2000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))


# replace . und \n
sentences = [sent.replace('\n',' ') for sent in sentences]
sentences = [sent.replace('.','') for sent in sentences]
sentences = [sent.encode('utf-8') for sent in sentences]

######### copy

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
vocabulary_size = len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
 
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

print "Training Data for RNN ready!"

##############################################################


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model parameters
            #save_model_parameters_theano("./saved_model/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    # ADDED! Saving model parameters
    save_model_parameters_theano("./saved_model/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)


def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while ( (not (new_sentence[-1] == word_to_index[sentence_end_token])) ):
        next_word_probs = model.forward_propagation(new_sentence)
        next_word_index = np.argmax(np.random.multinomial(1, next_word_probs[-1]))
        sampled_word = next_word_index
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    #sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    sentence_str = [index_to_word[x] for x in new_sentence]
    return sentence_str


#############################################################


# RNNLM


model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

if _MODEL_FILE != None:
	print "loading model"
	load_model_parameters_theano(_MODEL_FILE, model)

else:
	train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)
	print "new model trained with sgd"

#######################################################################################################################

# CREATING TEXT

num_sentences = 10
senten_min_length = 7

##############################################################

# BIGRAM generating sentences

text_list = sum(tokenized_sentences, [])
wordlist = getWords(text_list)
bigram = createBigram(text_list)

#bi_sentence = createSentence2(bilm, bigram, wordlist, 5)

print "Bigram created Sentences:"

bi_sentences = []
for i in range(num_sentences):
	sent = []
	# We want long sentences, not sentences with one or two words
	while len(sent) < senten_min_length:
		sent = createSentence2(bilm, bigram, wordlist)
	bi_sentences.append(sent)
	print " ".join(sent)

#print bi_sentences

##############################################################

# RNN-LM generating sentences

print "RNN-LM created Sentences:"

rnn_sentences = []
for i in range(num_sentences):
	sent = []
	# We want long sentences, not sentences with one or two words
	while len(sent) < senten_min_length:
		sent = generate_sentence(model)
	rnn_sentences.append(sent)
	print " ".join(sent)


