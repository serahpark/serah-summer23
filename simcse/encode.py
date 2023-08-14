from simcse import SimCSE
import pandas as pd

DATABASE_FILE = "corpus_sample/filtered_corpus_after.csv"


def load_body_texts():
    df = pd.read_csv(DATABASE_FILE, low_memory=False)
    result = []
    for email_id in range(len(df)):        
        if type(df.iloc[email_id]['body_text']) == str:
            result.append(df.iloc[email_id]['body_text'] + f"#{email_id}")
        else:
            result.append("")
    return result


all_texts = load_body_texts()

def find_identical_body(email_id):
    identicals = set()
    for i in range(len(all_texts)):
        if all_texts[i] == all_texts[email_id]:
            identicals.add(i)
    return identicals

def nonempty_rows():
    df = pd.read_csv(DATABASE_FILE)
    result = []
    for email_id in range(len(df)):        
        if type(df.iloc[email_id]['body_text']) == str:
            result.append(email_id)
    return result


#body_texts = load_body_texts()

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
model.build_index(all_texts)

def common_neighbors(sents):
    neighbor_sents = model.search(sents)
    return neighbor_sents


def nearest_neighbors(sents):
    neighbor_sents = model.search(sents)
    nearest = []
    for neighborhood in neighbor_sents:
        if len(neighborhood) <= 1:
            nearest.append(None)
        else:
            nearest.append(neighbor_sents[1])


def very_similar(sents, threshold=0.98):
    neighbor_sents = model.search(sents, threshold=threshold, top_k=10)
    all_buddies = set()
    for neighborhood in neighbor_sents:
        index = 0
        buddies = set()
        while index < len(neighborhood) and neighborhood[index][1] >= threshold:
            neighbor = neighborhood[index][0]
            neighbor_id = neighbor.split("#")[-1]
            buddies.add(neighbor_id)
            index += 1
        buddies = tuple(sorted(buddies))
        if len(buddies) > 1:
            all_buddies.add(buddies)
    return all_buddies

