# Multi-Resolution Time Series Transformer (MTST)

A PyTorch implementation of the Multi-Resolution Time-Series Transformer for Long-term Forecasting, based on the paper by Zhang et al. (2024). This model processes temporal data at different patch-level resolutions to capture both short-term and long-term patterns effectively.

## Overview

The MTST model employs a multi-resolution approach to time series forecasting by:
- Using **patch-level tokenization** with different patch sizes instead of timestamp-level tokenization
- Processing input sequences through multiple branches with different temporal resolutions
- Employing **relative positional encoding** to better capture periodic patterns
- Fusing multi-resolution features through concatenation and linear transformation
- Generating forecasts through learned linear projections

> **Note**: This implementation differs from the original paper by using stride-based subsampling for multi-resolution instead of different patch sizes, but achieves similar multi-scale temporal modeling.

## Architecture

```
Input Time Series (B, L, D)
           ↓
    ┌─────────────┬─────────────┬─────────────┐
    │  High-Res   │   Mid-Res   │   Low-Res   │
    │ (stride=1)  │ (stride=4)  │ (stride=10) │
    │      ↓      │      ↓      │      ↓      │
    │ ValueEnc    │ ValueEnc    │ ValueEnc    │
    │      ↓      │      ↓      │      ↓      │
    │ PosEnc      │ PosEnc      │ PosEnc      │
    │      ↓      │      ↓      │      ↓      │
    │Transformer  │Transformer  │Transformer  │
    └─────────────┴─────────────┴─────────────┘
           ↓             ↓             ↓
       Interpolation → Concatenation (3×embed_dim)
                  ↓
             Fusion Layer
                  ↓
             Linear Layers
                  ↓
             Forecast (B, output_len)
```

**Key Differences from Original Paper:**
- **Subsampling approach**: Uses stride-based subsampling (`x[:, ::stride, :]`) instead of different patch sizes
- **Interpolation fusion**: Aligns different resolution outputs to the same length before concatenation
- **Simplified architecture**: Focuses on the core multi-resolution concept with practical implementation choices

## Features

- **Multi-Resolution Processing**: Captures patterns at different temporal scales through stride-based subsampling
- **Transformer-Based**: Leverages attention mechanisms for sequence modeling
- **Interpolation Fusion**: Intelligent alignment and combination of multi-resolution representations
- **Flexible Architecture**: Configurable embedding dimensions, attention heads, and layers
- **Sinusoidal Positional Encoding**: Standard positional embeddings for temporal awareness
- **End-to-End Training**: Direct optimization for forecasting objectives

## Installation

```bash
git clone <repository-url>
cd mtst
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
import torch
from src.model.mtst import MTST

# Initialize model
model = MTST(
    input_dim=6,      # Number of input features
    embed_dim=64,     # Embedding dimension
    heads=8,          # Attention heads
    dropout=0.1,      # Dropout rate
    n_layers=10,      # Transformer layers
    output_len=5,     # Forecast horizon
    max_len=5000      # Max sequence length
)

# Example input: (batch_size, seq_len, input_dim)
x = torch.randn(32, 30, 6)

# Forward pass with resolution factors
forecast = model(x, high_res=1, mid_res=4, low_res=10)
print(f"Forecast shape: {forecast.shape}")  # [32, 5]
```

### Training

```python
# Run the complete training pipeline
python train.py
```

The training script includes:
- Data preprocessing and window creation
- Model initialization and training loop
- Loss tracking and visualization
- Model saving and inference on test data

### Data Format

Expected CSV format with columns:
- `High`, `Low`, `Open`, `Close`: OHLC price data
- `Volume`: Trading volume
- `Marketcap`: Market capitalization

## Model Parameters

### Core Parameters
- `input_dim`: Number of input features (default: 6)
- `embed_dim`: Embedding dimension (default: 64)
- `heads`: Number of attention heads (default: 8)
- `dropout`: Dropout rate (default: 0.1)
- `n_layers`: Number of transformer layers (default: 10)
- `output_len`: Forecast horizon (default: 5)
- `max_len`: Maximum sequence length for positional encoding (default: 5000)

### Resolution Factors
- `high_res`: High resolution stride (default: 1) - captures fine-grained patterns
- `mid_res`: Mid resolution stride (default: 4) - captures medium-term trends  
- `low_res`: Low resolution stride (default: 10) - captures long-term seasonality

### Training Hyperparameters
- `batch_size`: 32
- `epochs`: 100
- `learning_rate`: 5e-4
- `weight_decay`: 1e-5
- `input_len`: 30 (sequence length)

## File Structure

```
mtst/
├── src/
│   ├── model/
│   │   └── mtst.py          # Main MTST model
│   └── transformer.py       # Transformer block
├── utils/
│   ├── encoder.py           # Value and positional encoders
│   ├── data_processing.py   # Data loading and preprocessing
│   ├── window.py            # Moving window dataset creation
│   └── plots.py             # Visualization utilities
├── train.py                 # Training script
├── data/
│   ├── aave_train.csv       # Training data
│   └── aave_test.csv        # Test data
└── outputs/                 # Generated outputs
    ├── mtst_model.pth       # Saved model
    ├── training_loss.png    # Loss curve
    ├── prediction_plot.png  # Prediction visualization
    └── predictions.csv      # Prediction results
```

## Performance Expectations

Based on the original paper's results, MTST should demonstrate:
- **State-of-the-art performance** on standard benchmarks (ETT, Weather, Traffic, Electricity datasets)
- **Consistent improvements** over single-resolution transformers like PatchTST
- **Better long-term forecasting** due to multi-scale temporal modeling
- **Robust performance** across different prediction horizons (96, 192, 336, 720 steps)

The paper reports achieving rank-1 performance on 28 out of 28 test settings across 7 datasets and 4 prediction horizons.

## Loss Function

The model uses Mean Squared Error (MSE) loss by default, with support for Mean Absolute Error (MAE) loss as suggested in recent literature.

## Key Components

### 1. Multi-Resolution Processing
The model processes input sequences through stride-based subsampling:
- **High-resolution** (stride=1): Fine-grained temporal patterns, full sequence length
- **Mid-resolution** (stride=4): Medium-term trends, 1/4 sequence length
- **Low-resolution** (stride=10): Long-term patterns, 1/10 sequence length

### 2. Feature Fusion
Multi-resolution features are combined through:
- **Interpolation**: Aligns different resolution outputs to the same sequence length
- **Concatenation**: Combines aligned feature representations  
- **Linear fusion**: Projects concatenated features back to embedding dimension

### 3. Transformer Architecture
Each resolution uses a dedicated transformer encoder with:
- Multi-head self-attention
- Feed-forward networks
- Layer normalization and residual connections
- Standard PyTorch TransformerEncoder implementation

## Implementation Notes

### Differences from Original Paper

This implementation makes several practical adaptations while preserving the core multi-resolution concept:

1. **Subsampling Method**: Uses stride-based subsampling (`x[:, ::stride, :]`) instead of patch-based tokenization with different patch sizes
2. **Positional Encoding**: Uses standard sinusoidal positional encoding instead of relative positional encoding
3. **Fusion Strategy**: Employs interpolation to align sequence lengths before concatenation, rather than complex multi-branch patch fusion
4. **Architecture Simplification**: Focuses on core transformer blocks rather than custom attention mechanisms

### Bug Fixes Applied

- Fixed variable name duplication: `forecast = forecast = output.view(...)` → `forecast = output.view(...)`
- Ensured proper tensor reshaping for final output
- Added proper interpolation handling for different sequence lengths

## GPU Support

The model automatically detects and uses CUDA if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Customization

### Custom Loss Functions
```python
# MSE Loss (default)
criterion = nn.MSELoss()

# MAE Loss alternative
criterion = nn.L1Loss()
```

### Custom Data Processing
Modify `data_processing.py` to handle different data formats or preprocessing requirements.

### Architecture Modifications
Adjust model parameters in `train.py` or create custom configurations for different use cases.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{zhang2024mtst,
  title={Multi-resolution Time-Series Transformer for Long-term Forecasting},
  author={Zhang, Yitian and Ma, Liheng and Pal, Soumyasundar and Zhang, Yingxue and Coates, Mark},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2024},
  volume={238},
  pages={2024}
}
```
The data is from 
''' https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?select=coin_Aave.csv '''

## Acknowledgments

- Based on the paper "Multi-resolution Time-Series Transformer for Long-term Forecasting" by Zhang et al. (2024)
- Inspired by recent advances in patch-based transformer architectures for time series
- Built using PyTorch framework
- Thanks to the original authors for making their research publicly available