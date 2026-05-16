import pandas as pd

def load_real_dataset():

    df = pd.read_csv(
        "data/creditcard.csv"
    )

    return df