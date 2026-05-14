import torch
from torch import nn
import math


class SelfAttention(nn.Module):
    def __init__(self, vector_dim):
        super().__init__()
        self.vector_dim = vector_dim

        self.wq = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wk = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wv = nn.Linear(vector_dim, vector_dim, bias=False)
        self.wo = nn.Linear(vector_dim, vector_dim, bias=False)

    def compute_attention(self, q, k, v):
        d_k = self.vector_dim

        attention_score = q @ torch.transpose(k, -2, -1)
        attention_score = attention_score / math.sqrt(d_k)

        attention_score = torch.softmax(attention_score, dim=-1)

        values = attention_score @ v

        return values

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        values = self.compute_attention(q, k, v)
        out = self.wo(values)

        return out


class MultiHeadSelfAttention(SelfAttention):
    def __init__(self, vector_dim, n_heads):
        super().__init__(vector_dim)

        self.att_dim = vector_dim // n_heads
        self.n_heads = n_heads

    def reshape_for_multihead_attention(self, x):
        batch_size, seq_length, vector_dim = x.shape

        x = torch.reshape(
            x,
            (batch_size, seq_length, self.n_heads, self.att_dim),
        )

        x = torch.permute(x, (0, 2, 1, 3))

        return x

    def reshape_after_multihead_attention(self, x):
        batch_size, no_heads, seq_length, att_dim = x.shape
        vector_dim = att_dim * no_heads

        x = torch.permute(x, (0, 2, 1, 3))
        x = torch.reshape(x, (batch_size, seq_length, vector_dim))

        return x

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.reshape_for_multihead_attention(q)
        k = self.reshape_for_multihead_attention(k)
        v = self.reshape_for_multihead_attention(v)

        values = self.compute_attention(q, k, v)
        values = self.reshape_after_multihead_attention(values)

        out = self.wo(values)

        return out