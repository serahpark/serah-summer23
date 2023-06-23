import pandas as pd

def sample(path):
    n = 3
    df = pd.read_csv('small_fixed.csv', skiprows=lambda i: i % n != 0)

    df.to_csv(path)

if __name__ == "__main__":
    sample('test_sample.csv')