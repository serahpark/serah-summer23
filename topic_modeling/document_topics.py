"""This script iterates through each email (as an item in the corpus), gets the distribution
of the top topics, and adds the email's ID (according to the original dataframe) and the
assigned probability to a dictionary. The dictionary's ID corresponds to the index of the topic."""

import pandas as pd
import pickle
from time import time
from gensim.models import LdaModel
from gensim.test.utils import datapath
import yaml
print("imported modules")

with open('../configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

topics_start = time()

saved_model = datapath(config['model_datapath'])
model = LdaModel.load(saved_model) # use instead of pickle

# load corpus and dictionary
with open(config['corpus_pickle'], "rb") as c:
    corpus = pickle.load(c)
print("loaded LDA model, corpus, and dictionary")

# read corpus to include the email ID
with open(config['df_pickle'], "rb") as f:
    df = pickle.load(f)

# initialize dictionary with desired topics
top_topics = config['top_topics'] # add topic IDs manually
top_documents = {}
for id in top_topics:
    top_documents[id] = []
print("initialized dictionary for top topics")

# iterate through emails to find each top topic
for i in range(len(corpus)):
    email_id = df.at[i, "uid_email"]
    doc_topics = model.get_document_topics(corpus[i], minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
    for pair in doc_topics:
        topic_id = pair[0]
        probability = pair[1]
        if config['cutoff_by_threshold']:
            if topic_id in top_topics and probability >= config['threshold']:
                top_documents[topic_id].append((email_id, probability))
        else:
            if topic_id in top_topics:
                top_documents[topic_id].append((email_id, probability))
print("populated dictionary with documents")

# sort the documents corresponding to each topic and get the top 5 documents per topic
for id in top_topics:
    top_documents[id].sort(key=lambda probability: probability[1], reverse=True)
    if not config['cutoff_by_threshold']:
        top_documents[id] = top_documents[id][:config['num_docs']]
print(top_documents)

with open(config['docs_pickle'], "wb") as t:
    pickle.dump(top_documents, t)

topics_end = time()
print("Time to retrieve top documents per topic: " + str(topics_end - topics_start))