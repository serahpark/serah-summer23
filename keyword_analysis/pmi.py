import pandas as pd
import pickle
import numpy as np

def keyword_count(w):
    corpus = pd.read_csv('corpus_sample/primary_candidates_corpus.csv', index_col=0)
    count = 0
    for index, row in corpus.iterrows():
        if w in str(row['body_text']):
            count += 1
    return count

def cooccurrence(w):
    subset = pd.read_csv('corpus_sample/primary_candidates_crime_subset.csv', index_col=0)
    count = 0
    for index, row in subset.iterrows():
        if w in str(row['body_text']):
            count += 1
    return count


def pmi(corpus_path, subset_path, pickled_tokens):
    corpus = pd.read_csv(corpus_path, index_col=0)
    subset = pd.read_csv(subset_path, index_col=0)
    print('loaded csvs')

    corpus_len = len(corpus)
    subset_len = len(subset)
    p_t = subset_len / corpus_len

    with open(pickled_tokens, 'rb') as t:
        subset_tokens = pickle.load(t)
    print('loaded pickled tokens')

    original_case_w_count = {}
    for email in subset_tokens:
        for word in email:
            original_case_w_count[word] = 1 + original_case_w_count.get(word, 0)

    with open('pickle/original_case_w_count_dictionary', "wb") as d:
        pickle.dump(original_case_w_count, d)
    print("cooccurrence dictionary created & pickled as 'pickle/original_case_w_count_dictionary'")

    w_list = [w for w in original_case_w_count if original_case_w_count[w] > 20]
    pmi = pd.DataFrame(w_list, columns=['w'])
    print('dataframe initialized')

    pmi['corpus_count'] = pmi['w'].apply(keyword_count)
    print('w counts complete')
    pmi.to_csv('keyword_counts_original_case.csv')

    pmi['p(w)'] = pmi['corpus_count'] / corpus_len
    pmi['p(t,w)'] = pmi['w'].apply(cooccurrence)
    pmi['p(t,w)/p(t)p(w)'] = pmi['p_tw'] / (pmi['p_w'] * primary_p_t)

    pd.set_option('display.float_format', lambda x: '%0.5f' % x)
    pmi['pmi'] = np.log(pmi['p(t,w)/p(t)p(w)'])

    print('pmi calculated')
    
    pmi = pmi.sort_values('pmi', ascending=False)
    pmi.to_csv('og_case_pmi.csv')

if __name__ == "__main__":
    pmi('corpus_sample/primary_candidates_corpus.csv', 'corpus_sample/primary_candidates_crime_subset.csv', 'pickle/primary_candidates_crime_tokens_original_case')
