import pandas as pd
import numpy as np

def create_subsamp(df):
    # Shuffling the data before creating the sub-samples
    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows again
    new_df = normal_distributed_df.sample(frac=1, random_state=42)
    return new_df