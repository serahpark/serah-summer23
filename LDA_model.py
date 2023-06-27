import pandas as pd
import spacy
import pickle
from time import time
from spacy.language import Language
print("imported modules")

# Opening the pre-processed corpus data
with open("0626_dictionary", "rb") as d:
    dictionary = pickle.load(d)   
with open("0626_corpus", "rb") as c:
    corpus = pickle.load(c)
print("unpickled corpus and dictionary files")

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Enable logging to see the progress of training
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print("logging setup successful")

lda_start = time()

# Train LDA model.
from gensim.models import LdaModel

# Set training parameters.
num_topics = 100
chunksize = 100000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

lda_end = time()
print("LDA run time: " + str(lda_end - lda_start))

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)
print("LDA on full corpus complete!")