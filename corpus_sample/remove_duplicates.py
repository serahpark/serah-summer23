import pandas as pd
from random import choice
print('imported modules')

def remove_duplicates(file_path):
    # load csv as dataframe
    df = pd.read_csv(file_path)
    print(len(df))
    # drop non-candidates - remove entries with non-empty type?
    df = df.loc[df['type'].isna()]
    print('len after filter by type')
    print(len(df))
    # identify all unique from_address emails & their counts
    email_addresses = df.groupby(['from_address']).count().sort_values('body_text', ascending=False).index
    # print(email_addresses[:5])

    # for every from_address with more than one email, make a list of uid_inbox, randomly choose one, and drop all other entries without that uid_inbox
    for email in email_addresses:
        inbox_ids = set()
        unique_sender_emails = df.loc[df['from_address'] == email]
        for index, row in unique_sender_emails.iterrows():
            inbox_ids.add(row['uid_inbox'])
        select_id = choice(list(inbox_ids))
        df = df.loc[~((df['from_address'] == email) & (df['uid_inbox'] != select_id))]
        
    # df.to_csv
    print('len after filtering top 10 senders')
    print(len(df))
    df.to_csv('filtered_corpus.csv')

if __name__ == '__main__':
    remove_duplicates('~/../data/princeton_emails/corpus_v1.0.csv')