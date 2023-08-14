import pandas as pd
from remove_duplicates import *

def filter_by_time(date, before=True):
    # use if corpus has not been filtered for candidates-only yet
    #df = remove_duplicates('~/../data/princeton_emails/corpus_v1.0.csv')    
    
    df = pd.read_csv('filtered_corpus.csv')

    if before:
        before_date = df.loc[df['date'] < date]
        before_date.to_csv('filtered_corpus_before.csv', index=False)
    else:
        after_date = df.loc[df['date'] >= date]
        after_date.to_csv('filtered_corpus_after.csv', index=False)


if __name__ == '__main__':
    filtered = filter_by_time('2020-05-25')