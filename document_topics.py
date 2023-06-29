"""This script iterates through each email (as an item in the corpus), gets the distribution
of the top topics, and adds the email's ID (according to the original dataframe) and the
assigned probability to a dictionary. The dictionary's ID corresponds to the index of the topic."""

import pandas as pd
import pickle
from time import time
from gensim.models import LdaModel
from gensim.test.utils import datapath
print("imported modules")

topics_start = time()

model = LdaModel.load(saved_model) # use instead of pickle

# load corpus and dictionary
with open("0628_model", "rb") as m:
    model = pickle.load(m)
with open("pickle/0628_corpus", "rb") as c:
    corpus = pickle.load(c)
print("loaded LDA model, corpus, and dictionary")

# read corpus to include the email ID
with open("pickle/0628_df", "rb") as f:
    df = pickle.load(f)
#df = pd.read_csv('~/../data/princeton_emails/corpus_v1.0.csv', usecols=[0, 1, 2, 3, 18]).dropna().reset_index(drop=True)

# initialize dictionary with desired topics
top_topics = [3, 7, 9, 12, 29, 71, 96] # add topic IDs manually
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

with open("pickle/0628_docs_postsave", "wb") as t:
    pickle.dump(top_documents, t)

topics_end = time()
print("Time to retrieve top documents per topic: " + str(topics_end - topics_start))