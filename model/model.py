import torch
import torch.nn as nn
import torch.nn.functional as F
from .fully_connected_layer import FullyConnectedLayer
from .multi_head_attention import MultiHeadAttention

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.swish_activation = nn.SiLU()
        self.layernorm = nn.LayerNorm(d_model)
        self.linear1 = FullyConnectedLayer(d_model, d_model * 4)
        self.linear2 = FullyConnectedLayer(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.linear1(x)
        x = self.swish_activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        # we scale the output by 0.5 acroding to the diagramm in the presentation 
        # Residual connections are added onto the ouput and provide a direct path for gradients 
        return residual + 0.5 * x



class AttentionBlock(nn.Module):
    # Wraps the self-attention with pre-LayerNorm, residual and dropout
    # (the MultiHeadAttention class itself only does the raw attention).
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.layernorm(x)
        x = self.mha(x, mask)
        x = self.dropout(x)
        # full residual (no 0.5 scaling like the FeedForwardBlock)
        return residual + x


class ConvolutionalBlock(nn.Module):
    # Convolutional Block(see Slides)
    def __init__(self, d_model, dropout, conv_channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=2*conv_channels, kernel_size=1)
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv1 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, groups=conv_channels, padding=1)
        self.BatchNorm = nn.BatchNorm1d(conv_channels)
        self.swish_activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = x.transpose(1, 2)     
        x = self.pointwise_conv1(x)
        x = self.glu_activation(x)
        x = self.depthwise_conv1(x) 
        x = self.BatchNorm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)     
        x = self.dropout(x)
        return residual + x
    
class ConformerBlock(nn.Module):
    # Conformer Block(see Slides)
    def __init__(self, d_model, dropout, num_heads=4, conv_channels=31, use_attention=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.FeedForwardBlock1 = FeedForwardBlock(d_model, dropout)
        self.use_attention = use_attention
        if use_attention:
            self.attentionBlock = AttentionBlock(d_model, num_heads, dropout)
        self.FeedForwardBlock2 = FeedForwardBlock(d_model, dropout)
        self.convolutionalBlock = ConvolutionalBlock(d_model, dropout, conv_channels)


    def forward(self, x, mask=None):
        x = self.FeedForwardBlock1(x)
        # Conformer order: FFN½ -> MHA -> Conv -> FFN½ -> LayerNorm
        if self.use_attention:
            x = self.attentionBlock(x, mask)
        x = self.convolutionalBlock(x)
        x = self.FeedForwardBlock2(x)
        x = self.layernorm(x)
        return x


class Frontend(nn.Module):
    # We reduce the time dimension from 40 to 10 and project to d_model=100 features per timestep.
    def __init__(self, d_model, n_mels=80, kernel_size=3, pool_kernel=(4, 1)):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(16, 16, kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        self.linear = FullyConnectedLayer(16 * n_mels, d_model)

    def forward(self, x):
        # x arrives as (batch, time, n_mels)
        # We add a channel axis at position 1 for the Conv2d layers(See documentation nn.Conv2d)
        x = x.unsqueeze(1)                    # (batch, 1, time, n_mels) — add channel axis
        
        x = self.activation(self.conv1(x))  
        x = self.activation(self.conv2(x))  
        #Max pool reduces the time dimension by 4
        x = self.pool(x)                     
        
        # Conv2d outputs (batch, channels, time, freq) but we need (batch, time, features) for the Conformer blocks so we
        # rearrange the axes so that the channels are next to the n_mels and then flatten them together into one dimesion
        x = x.permute(0, 2, 1, 3)           
        x = x.flatten(2)                      
        
        x = self.linear(x)              
        
        return x


class Conformer(nn.Module):
    def __init__(self, d_model, num_blocks, vocab_size, dropout, n_mels=80, conv_channels=31, num_heads=4, use_attention=True):
        super().__init__()
        self.frontend = Frontend(d_model, n_mels)
        self.projection = FullyConnectedLayer(d_model, d_model)  # optional but common
        self.pos_encoding = nn.Embedding(5000, d_model)  # learned positions
        self.dropout = nn.Dropout(dropout)
        self.use_attention = use_attention
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, dropout, num_heads, conv_channels, use_attention) for _ in range(num_blocks)
        ])
        self.output_linear = FullyConnectedLayer(d_model, vocab_size)

    def forward(self, x, input_lengths=None):
        x = self.frontend(x)        # (batch, time/4, d_model)
        x = self.projection(x)
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_encoding(positions)
        x = self.dropout(x)

        # Build a padding mask so self-attention ignores padded frames.
        # input_lengths are the real lengths AFTER the /4 frontend pooling.
        mask = None
        if self.use_attention and input_lengths is not None:
            T = x.size(1)
            valid = torch.arange(T, device=x.device)[None, :] < input_lengths[:, None]
            mask = valid[:, None, None, :]   # (batch, 1, 1, T'), broadcasts over heads/queries

        for block in self.blocks:
            x = block(x, mask)
        x = self.output_linear(x)   # (batch, time/4, vocab_size)
        return x
    
    
