import pandas as pd
import numpy as np
import torch

def load_data(file_path):
    """
    Load time series data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file containing the time series data.
        
    Returns:
        pd.DataFrame: DataFrame containing the loaded time series data.
    """
    try:
        data = pd.read_csv(file_path, parse_dates=True, index_col=0)
        remove_col = ['SNo', 'Name', 'Symbol', 'Date']
        data.drop(columns=[col for col in remove_col], inplace=True, errors='ignore')
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def preprocess_data(data, fillna_method='ffill'):
    """
    Preprocess time series data by handling missing values and normalizing.
    
    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        fillna_method (str): Method to fill missing values ('ffill', 'bfill', 'mean', etc.).
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    if fillna_method == 'ffill':
        data.fillna(method='ffill', inplace=True)
    elif fillna_method == 'bfill':
        data.fillna(method='bfill', inplace=True)
    elif fillna_method == 'mean':
        data.fillna(data.mean(), inplace=True)
    
    return data

def create_moving_window_dataset(data, input_len, output_len, stride=1):
    """
    Create a moving window dataset from the time series data.
    
    Args:
        data (pd.DataFrame): DataFrame containing the time series data.
        input_len (int): Length of the input sequence.
        output_len (int): Length of the output sequence.
        stride (int): Step size for moving the window.
        
    Returns:
        np.ndarray: Input sequences.
        np.ndarray: Output sequences.
    """
    x, y = [], []
    
    for start in range(0, len(data) - input_len - output_len + 1, stride):
        end = start + input_len
        x.append(data.iloc[start:end].values)
        y.append(data.iloc[end:end + output_len].values)
    
    return np.array(x), np.array(y)

def final_data_processing(file_path, input_len=120, output_len=24, stride=1, fillna_method='ffill'):
    """
    Load and preprocess time series data, then create a moving window dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the time series data.
        input_len (int): Length of the input sequence.
        output_len (int): Length of the output sequence.
        stride (int): Step size for moving the window.
        fillna_method (str): Method to fill missing values ('ffill', 'bfill', 'mean', etc.).
        
    Returns:
        np.ndarray: Input sequences.
        np.ndarray: Output sequences.
    """
    data = load_data(file_path)
    if data is None:
        return None, None
    
    data = preprocess_data(data, fillna_method)
    x_data, y_data = create_moving_window_dataset(data, input_len, output_len, stride)
    
    return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)

