import pandas as pd
import spacy
import pickle
from time import time
from spacy.language import Language
print("imported modules")

# loading spacy & adding modifications to the pipeline and list of stopwords
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")
nlp.add_pipe("merge_entities")
nlp.Defaults.stop_words |= {"alex", "unsubscribe"}
print("loaded spacy model & added modifications")

preprocess_start = time()

# reading in csv as a pandas dataframe
df = pd.read_csv('../../data/princeton_emails/corpus_v1.0.csv', usecols=range(4)).dropna()
print("successfully read in csv")

# extracting body text of email
body_text = df.loc[:, 'body_text']

# piping documents through nlp()
emails = []
include_named_entities = True
print("ready to iterate through emails")

for email in list(nlp.pipe(body_text)):
    if include_named_entities:
        emails.append([token.lemma_.lower() for token in email if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2])
    else:
        emails.append([token.lemma_.lower() for token in email if not token.is_stop and not token.is_punct and not token.like_num and not token.ent_type_ and len(token.text) > 2])
print("spacy preprocessing complete!")

import gensim
import gensim.corpora as corpora

print("completing gensim steps - creating dictionary")
# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(emails)

# Filter out words that occur less than 10 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=3000, no_above=0.25)

# Bag-of-words representation of documents
corpus = [dictionary.doc2bow(doc) for doc in emails]
print("completed corpus of BOW generated")
print(corpus[0])

# storing dictionary and corpus as pickled files
with open("0626_dictionary", "wb") as d:
    pickle.dump(dictionary, d)

with open("0626_corpus", "wb") as c:
    pickle.dump(corpus, c)

preprocess_end = time()
print("Time to pre-process corpus: " + str(preprocess_end - preprocess_start))