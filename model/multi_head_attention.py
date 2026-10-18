import torch
import math 
import torch.nn as nn
import torch.nn.functional as F
from . import fully_connected_layer
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)
        self.scaling_factor = math.sqrt(self.head_dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.W_q = fully_connected_layer.FullyConnectedLayer(d_model, d_model)
        self.W_k = fully_connected_layer.FullyConnectedLayer(d_model, d_model)
        self.W_v = fully_connected_layer.FullyConnectedLayer(d_model, d_model)
        self.W_o = fully_connected_layer.FullyConnectedLayer(d_model, d_model)

    def ScaledDotProductAttention(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling_factor

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # nan_to_num zeros out rows that were all -inf (fully-padded sequences)
        attention_weights = self.dropout(F.softmax(scores, dim=-1).nan_to_num(0.0))
        output = torch.matmul(attention_weights, V)

        return output
    
    def forward(self, x, mask=None):
        # Instead of Weight matrices we use a linear layer to project into Q,K,V since theyre equivalent
        batch_size, seq_len, _ = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # We reshape Q,K,V to (batch_size, num_heads, seq_len, head_dim), adding a new dimesnion, 
        # so we can perform attention for each head in parallel.
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
	    #QK-Norm
        Q = self.q_norm(Q)
        K = self.k_norm(K)

        output = self.ScaledDotProductAttention(Q, K, V, mask)
        # We flatten the dimensions back 
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(output)