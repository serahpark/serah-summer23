from sys import argv
import pandas as pd

def check_sample(path):
    df = pd.read_csv(path, usecols=[1,2,3,4], nrows=10)
    return df

if __name__ == "__main__":
    path = argv[1]
    print(check_sample(path))

