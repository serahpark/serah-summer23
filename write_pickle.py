import pickle
import yaml

with open('topic_modeling.yaml', 'r') as file:
     config = yaml.safe_load(file)
with open(config['docs_pickle'], "rb") as d:
    topics = pickle.load(d)

with open('0716_docs.txt', 'w') as writer:
    writer.write(str(topics))