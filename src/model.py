import torch
from torch import nn
from attention import MultiHeadSelfAttention


"""
    How the flow of this model works?
    - Input: example review "This movie was amazing" → gets converted into token IDs
    - Embeddings: self.token_embedding → turns token IDs into vectors
    - Positional Embeddings: self.position_embedding → adds word position information
    - Transformer blocks: self.blocks → learns relationships between words using attention
    - Mean pooling: pooled = x.sum(dim=1)/... → turns all token vectors into ONE sentence vector
    - Final classifier: self.classifier → outputs [negative_score, positive_score]
"""

""" 
    one transformer layer
    Self-attention
    → residual connection
    → layer norm
    → feed-forward network
    → residual connection
    → layer norm
"""

class TransformerBlock(nn.Module):
    def __init__(self, vector_dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(vector_dim, n_heads)
        self.norm1 = nn.LayerNorm(vector_dim)
        self.norm2 = nn.LayerNorm(vector_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vector_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_out = self.attention(x)
        x = self.norm1(x + self.dropout(attention_out))

        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


""" 
    the full model. non-casual model; meaning that every word can attend to every other word
    Input tokens
    → token embeddings
    → positional embeddings
    → multiple Transformer blocks
    → mean pooling
    → classifier
    → positive/negative output
"""
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        vector_dim=128,
        n_heads=4,
        n_layers=2,
        hidden_dim=256,
        dropout=0.1,
        num_classes=2,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, vector_dim)
        self.position_embedding = nn.Embedding(block_size, vector_dim)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    vector_dim=vector_dim,
                    n_heads=n_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.classifier = nn.Linear(vector_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        x = self.blocks(x)

        mask = attention_mask.unsqueeze(-1)
        x = x * mask

        # turns all token vectors into ONE sentence vector
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled)

        return logits