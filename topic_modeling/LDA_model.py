import pandas as pd
import spacy
import pickle
from time import time
from spacy.language import Language
from gensim.models import LdaModel
import logging
import yaml
print("imported modules")

with open('../configs.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Opening the pre-processed corpus data
with open(config['dict_pickle'], "rb") as d:
    dictionary = pickle.load(d)   
with open(config['corpus_pickle'], "rb") as c:
    corpus = pickle.load(c)
print("unpickled corpus and dictionary files")

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

# Enable logging to see the progress of training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print("logging setup successful")

lda_start = time()

# Set training parameters.
num_topics = config['num_topics']
chunksize = 100000
passes = config['num_passes']
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


from gensim.test.utils import datapath
saved_model = datapath(config['model_datapath'])
model.save(saved_model)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)
print("LDA on full corpus complete!")

with open(config['topic_output'], 'w') as writer:
    for i in range(100):
        writer.write(f"TOPIC {i}:\n")
        writer.write(model.print_topic(i, topn=20))
        writer.write("\n")

# # DOCUMENT TOPICS BEFORE SAVING
# topics_start = time()

# # read corpus to include the email ID
# with open(config['df_pickle'], "rb") as f:
#     df = pickle.load(f)


# # initialize dictionary with desired topics
# interesting_topics = config['top_topics'] # add topic IDs manually
# top_documents = {}
# for id in top_topics:
#     top_documents[id] = []
# print("initialized dictionary for top topics")

# # iterate through emails to find each top topic
# assert len(corpus) == df.shape[0]
# for i in range(len(corpus)):
#     email_id = df.at[i, "uid_email"]
#     doc_topics = model.get_document_topics(corpus[i], minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
#     for pair in doc_topics:
#         topic_id = pair[0]
#         probability = pair[1]
#         if topic_id in top_topics:
#             top_documents[topic_id].append((email_id, probability))
# print("populated dictionary with documents")

# # sort the documents corresponding to each topic and get the top 5 documents per topic
# for id in top_topics:
#     top_documents[id].sort(key=lambda probability: probability[1], reverse=True)
#     top_documents[id] = top_documents[id][0:5]
# print(top_documents)

# with open(config['presave_pickle'], "wb") as t:
#     pickle.dump(top_documents, t)

# topics_end = time()
# print("Time to retrieve top documents per topic: " + str(topics_end - topics_start))

