import torch
import torch.nn as nn

class ValueEncoder(nn.Module):
    """
    Value Encoder for Transformer models.
    This module encodes input values into a higher-dimensional space.
    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features. Default is 64.
    Returns:
        torch.Tensor: Encoded features of shape (batch_size, output_dim).
    """
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class PositionalEncoder(nn.Module):
    """
    Positional Encoding for Transformer models.
    This module adds positional information to the input embeddings.
    This is an implementation of the sinusoidal positional encoding as described in the Transformer paper.
    Args:
        embed_dim (int): Dimension of the embeddings.
        max_len (int): Maximum length of the input sequences.
    
    Returns:
        torch.Tensor: Positional encoded embeddings.
    """
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)