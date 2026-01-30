import pandas as pd
import os

def load_data(filepath: str = "corpus/raw_prices.csv") -> pd.DataFrame:

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. "
                                f"Run `python corpus/synth_forge.py` to generate it.")
    
    df = pd.read_csv(filepath)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "price"):

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

