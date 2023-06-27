"""This script generates a systematic sample of a longer document, ex. corpus_v1.0.csv, by taking every nth line in the document.
In the command line, provide two arguments: the number of lines needed for the sample, and the path to the new file."""

import pandas as pd
from sys import argv

# length of original document
file_len = 310000

def sample(num_lines, path, file_name):
    n = file_len // int(num_lines)
    df = pd.read_csv(file_name, skiprows=lambda i: i % n != 0 if i <= n * num_lines else True)#, usecols=range(4)).dropna()

    df.to_csv(path)

if __name__ == "__main__":
    # takes command line arguments as inputs
    num_lines = int(argv[1])
    path = argv[2]
    sample(num_lines, path, '~/../data/princeton_emails/corpus_v1.0.csv')
    # print(num_lines)
    # print(path)