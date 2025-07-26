import torch
import torch.nn as nn
from utils.encoder import ValueEncoder, PositionalEncoder
from src.transformer import TransformerBlock

class MTST(nn.Module):
    """
    Multi-Res Transformer for Time Series Forecasting.
    This model uses a multi-resolution approach to process time series data at different resolutions
    and combines the results for forecasting.
    Args:
        input_dim (int): Dimension of the input features.
        embed_dim (int): Dimension of the embedding space. Default is 64.
        heads (int): Number of attention heads. Default is 10.
        dropout (float): Dropout rate. Default is 0.1.
        n_layers (int): Number of transformer layers. Default is 5.
        output_len (int): Length of the output forecast. Default is 24.
        max_len (int): Maximum length of the input sequences for positional encoding. Default is 5000.
    Returns:
        torch.Tensor: Forecasted values of shape (batch_size, output_len).
    """
    def __init__(self, input_dim, embed_dim=64, heads=10, dropout=0.1, n_layers=5, output_len=24, max_len=5000):
        super().__init__()
        self.value_encoder = ValueEncoder(input_dim, embed_dim)
        self.pe_high = PositionalEncoder(embed_dim, max_len)
        self.pe_mid = PositionalEncoder(embed_dim, max_len)
        self.pe_low = PositionalEncoder(embed_dim, max_len)
        self.transformer_high_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.transformer_mid_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.transformer_low_res = TransformerBlock(embed_dim, heads, dropout, n_layers)
        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, 1)
        self.output_layer = nn.Linear(1, output_len)
        self.relu = nn.ReLU()
    
    def interpolate(self, x, target_size):
        """
        Interpolates the input tensor to the target size.
        Input shape: (batch_size, seq_len, embed_dim)
        Output shape: (batch_size, target_size, embed_dim)
        """
        x = x.permute(0, 2, 1)  # (B, embed_dim, seq_len)
        x = nn.functional.interpolate(x, size=target_size, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)  # (B, target_size, embed_dim)
        return x

    
    def forward(self, x, high_res, mid_res, low_res):
        """
        Forward pass of the MTST model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            high_res (int): High resolution factor.
            mid_res (int): Mid resolution factor.
            low_res (int): Low resolution factor.
        Returns:
            torch.Tensor: Forecasted values of shape (batch_size, output_len).
        """
        x_high = x[:, ::high_res, :]
        x_mid = x[:, ::mid_res, :]
        x_low = x[:, ::low_res, :]

        def process(xr, transformer, positional_encoder):
            """
            Process the input tensor through value encoding, positional encoding, and transformer layers.
            Args:
                xr (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
                transformer (TransformerBlock): Transformer block to apply.
                positional_encoder (PositionalEncoder): Positional encoder to apply.
            Returns:
                torch.Tensor: Processed tensor of shape (batch_size, seq_len, embed_dim).
            """
            xr = self.value_encoder(xr)
            xr = positional_encoder(xr)
            xr = transformer(xr)
            return xr
        
        x_high = process(x_high, self.transformer_high_res, self.pe_high)
        x_mid = self.interpolate(process(x_mid, self.transformer_mid_res, self.pe_mid), x_high.size(1))
        x_low = self.interpolate(process(x_low, self.transformer_low_res, self.pe_low), x_high.size(1))

        fused = torch.cat((x_high, x_mid, x_low), dim=-1)
        fused = self.fusion_layer(fused)
        fused = torch.mean(fused, dim=1) # Global average pooling. This vector summarizes all time steps into one fixed-length embedding.
        fused = self.linear1(fused)
        fused = torch.relu(fused)
        fused = self.linear2(fused)
        fused = torch.relu(fused)
        forecast = self.output_layer(fused)
        return forecast