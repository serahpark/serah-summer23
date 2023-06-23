"""This script generates a systematic sample of a longer document, ex. corpus_v1.0.csv, by taking every nth line in the document.
In the command line, provide two arguments: the number of lines needed for the sample, and the path to the new file."""

import pandas as pd
from sys import argv

# length of original document
file_len = 300000

def sample(num_lines, path, file_name):
    # n represents the intervals between each line being sampled; set to 300k for corpus estimate
    n = file_len // int(num_lines)

    # reads the first nrows entries of the csv and skips n rows for each sampled line
    # df = pd.read_csv(file_name, nrows=n*num_lines, skiprows=lambda i: i % n != 0)
    df = pd.read_csv(file_name, skiprows=lambda i: i % n != 0 if i <= n * num_lines else True, usecols=range(4)).dropna()
    # directs sampled lines to the provided path
    df.to_csv(path)

if __name__ == "__main__":
    
    # takes command line arguments as inputs
    num_lines = int(argv[1])
    path = argv[2]
    sample(num_lines, path, 'corpus_v1.0.csv')