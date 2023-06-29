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
df = pd.read_csv('../../data/princeton_emails/corpus_v1.0.csv', usecols=[0, 1, 2, 3, 18]).dropna()
with open("pickle/0628_df", "wb") as f:
    pickle.dump(df, f)

print("successfully read in and pickled csv")
# save this df (with email id) so the same df can be used for doc_topics

# extracting body text of email
body_text = df.loc[:, 'body_text']

# piping documents through nlp()
emails = []
include_named_entities = True
print("ready to iterate through emails")

for email in body_text:
    email = nlp(email.replace('\n', ' '))
    if include_named_entities:
        emails.append([token.lemma_.lower() for token in email if not token.is_stop and token.text.isalpha() and len(token.text) > 2])
    else:
        emails.append([token.lemma_.lower() for token in email if not token.is_stop and token.text.isalpha() and not token.ent_type_])
print("spacy preprocessing complete!")

import gensim
import gensim.corpora as corpora

print("completing gensim steps - creating dictionary")
# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(emails)

# Filter out words that occur less than 3000 documents, or more than 75% of the documents.
dictionary.filter_extremes(no_below=3000, no_above=0.75)

# Bag-of-words representation of documents
corpus = [dictionary.doc2bow(doc) for doc in emails]
print("completed corpus of BOW generated")
print(corpus[0])

# storing dictionary and corpus as pickled files
with open("pickle/0628_dictionary", "wb") as d:
    pickle.dump(dictionary, d)

with open("pickle/0628_corpus", "wb") as c:
    pickle.dump(corpus, c)

preprocess_end = time()
print("Time to pre-process corpus: " + str(preprocess_end - preprocess_start))