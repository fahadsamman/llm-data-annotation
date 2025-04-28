import os
import pandas as pd

class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """Load a dataset from a given path and validate required columns exist."""
    if not os.path.exists(filepath):
        raise DataLoaderError(f"File does not exist: {filepath}")

    df = pd.read_csv(filepath)

    required_columns = {"id", "text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise DataLoaderError(f"Missing required columns: {missing_columns}")

    return df
