import pandas as pd
import spacy
import pickle
from time import time
from spacy.language import Language
import yaml
print("imported modules")

with open('topic_modeling.yaml', 'r') as file:
    config = yaml.safe_load(file)

# loading spacy & adding modifications to the pipeline and list of stopwords
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_noun_chunks")
nlp.add_pipe("merge_entities")
nlp.Defaults.stop_words |= {"alex", "unsubscribe", "url", "redacted", "url_redacted"}
print("loaded spacy model & added modifications")

preprocess_start = time()

# reading in csv as a pandas dataframe
df = pd.read_csv(config['subset']).dropna().reset_index(drop=True)
with open(config['subset_df_pickle'], "wb") as f:
    pickle.dump(df, f)

print("successfully read in and pickled csv")

# extracting body text of email
body_text = df.loc[:, 'body_text']

# piping documents through nlp()
emails = []
print("ready to iterate through emails")

for email in body_text:
    email = nlp(email.replace('\n', ' '))
    # if config['include_named_entities']:
    emails.append([token.text.lower() for token in email if not token.is_stop and not token.is_punct and not token.like_num and len(token.text) > 2])
    # else:
    #     emails.append([token.text.lower() for token in email if not token.is_stop and not token.is_punct and not token.like_num and not token.ent_type_])

print("spacy preprocessing complete!")
preprocess_end = time()
print('Time to preprocess subset: ' + str(preprocess_end - preprocess_start))

with open(config['tokens_pickle'], "wb") as t:
    pickle.dump(emails, t)

