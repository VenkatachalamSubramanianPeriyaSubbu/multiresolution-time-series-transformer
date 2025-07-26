import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads=10, dropout=0.1, n_layers=5):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.n_layers = n_layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x):
        return self.transformer(x)