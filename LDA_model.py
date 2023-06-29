import pandas as pd
import spacy
import pickle
from time import time
from spacy.language import Language
print("imported modules")

# Opening the pre-processed corpus data
with open("pickle/0628_dictionary", "rb") as d:
    dictionary = pickle.load(d)   
with open("pickle/0628_corpus", "rb") as c:
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

# DOCUMENT TOPICS BEFORE SAVING

# read corpus to include the email ID
with open("pickle/0628_df", "rb") as f:
    df = pickle.load(f)

# initialize dictionary with desired topics
top_topics = [3, 7, 9, 12, 29, 71, 96] # add topic IDs manually
top_documents = {}
for id in top_topics:
    top_documents[id] = []
print("initialized dictionary for top topics")

# iterate through emails to find each top topic
assert len(corpus) == df.shape[0]
for i in range(len(corpus)):
    email_id = df.at[i, "uid_email"]
    doc_topics = model.get_document_topics(corpus[i], minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    for pair in doc_topics:
        topic_id = pair[0]
        probability = pair[1]
        # can try with toy model to see that the topics are being generated correctly, also retry with before & after saving the model, and sanity check first few emails
        # add email to dictionary if desired topics are represented in the text
        if topic_id in top_topics:
            top_documents[topic_id].append((email_id, probability))
print("populated dictionary with documents")

# sort the documents corresponding to each topic and get the top 5 documents per topic
for id in top_topics:
    top_documents[id].sort(key=lambda probability: probability[1], reverse=True)
    top_documents[id] = top_documents[id][0:5]
print(top_documents)

with open("pickle/0628_docs_presave", "wb") as t:
    pickle.dump(top_documents, t)

topics_end = time()
print("Time to retrieve top documents per topic: " + str(topics_end - topics_start))

from gensim.test.utils import datapath
saved_model = datapath("0628_model")
model.save(saved_model)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)
print("LDA on full corpus complete!")

