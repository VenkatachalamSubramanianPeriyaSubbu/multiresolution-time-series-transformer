import torch

def create_moving_window_dataset(series, input_len, output_len, stride=1):
    """
    Create a moving window dataset from a time series.
    Args:
        series (torch.Tensor): Time series data of shape [T, D] or [T].
        input_len (int): Length of the input sequence.
        output_len (int): Length of the output sequence to predict.
        stride (int): Step size for moving window.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    """
    if series.ndim == 1:
        series = series.unsqueeze(1)
    T, D = series.shape
    X, Y = [], []

    for i in range(0, T - input_len - output_len + 1, stride):
        x_i = series[i : i + input_len]
        y_i = series[i + input_len : i + input_len + output_len, 0]
        X.append(x_i)
        Y.append(y_i)

    return torch.stack(X), torch.stack(Y)
