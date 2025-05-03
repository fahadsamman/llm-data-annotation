import os
import pandas as pd

class DataLoaderError(Exception):
    """Custom exception for data loading errors."""
    pass

def load_and_validate_data(filepath: str, id_column: str, text_column: str):
    """Load a dataset from a given path and validate required columns exist."""
    if not os.path.exists(filepath):
        raise DataLoaderError(f"File does not exist: {filepath}")

    df = pd.read_csv(filepath)

    if 'id' not in df.columns:
        if id_column is None:
            raise DataLoaderError("No id column specified and no id_column provided in config.")
        else:
            df['id'] = df[id_column].astype(str)

    if 'text' not in df.columns:
        if text_column is None:
            raise DataLoaderError("No text column specified and no text_column provided in config.")
        else:
            df['text'] = df[text_column].astype(str)

    # this part might be redundant now:
    required_columns = {"id", "text"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise DataLoaderError(f"Missing required columns: {missing_columns}")

    return df
