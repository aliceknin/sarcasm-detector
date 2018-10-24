import numpy
import nltk
from nltk.corpus import sentiwordnet


posproc = numpy.load('posproc.npy')
negproc = numpy.load('negproc.npy')

sarcastic_tweets = []
non_sarcastic_tweets = []

for tweet in posproc:
    sarcastic_tweets.append(tweet.decode('utf-8'))

for tweet in negproc:
    non_sarcastic_tweets.append(tweet.decode('utf-8'))

print(sarcastic_tweets[1])
print(non_sarcastic_tweets[0])

unigrams = {}

# for tweet in sarcastic_tweets


# print(sentiwordnet.senti_synset('breakdown.n.03').pos_score())






