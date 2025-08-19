import pandas as pd
import os

def load_data(filepath: str = "data/dataset.csv") -> pd.DataFrame:
    """
    Load the real estate dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with features and target (price).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. "
                                f"Run `python data/generate_data.py` to generate it.")
    
    df = pd.read_csv(filepath)
    return df


def split_features_target(df: pd.DataFrame, target_col: str = "price"):
    """
    Split the dataset into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and target.
    target_col : str
        Column name for the target variable.

    Returns
    -------
    X : pd.DataFrame
        Features (all columns except target).
    y : pd.Series
        Target variable (price).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
