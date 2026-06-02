import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

def load_and_scale_data(file_path):
    df = pd.read_csv(file_path)
    
    # RobustScaler is less prone to outliers
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Move scaled columns to the front for clarity
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)
    
    return df

def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    # Stratified split ensures the 0.172% ratio is preserved in both sets
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)