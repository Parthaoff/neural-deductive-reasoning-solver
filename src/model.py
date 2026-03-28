# src/model.py

import torch
import torch.nn as nn


class TransformerEncoderLayerWithAttention(nn.Module):
    """
    Custom Transformer encoder layer that returns attention weights.
    """

    def __init__(self, d_model, n_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self attention with attention weights
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False
        )

        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class LogicTransformer(nn.Module):
    """
    Transformer-based neural reasoning model.
    """

    def __init__(self,
                 vocab_size,
                 d_model=256,
                 n_heads=8,
                 num_layers=6,
                 num_classes=3,
                 max_len=64,
                 dropout=0.1):

        super().__init__()

        self.d_model = d_model
        self.max_len = max_len

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.position_embedding = nn.Embedding(max_len, d_model)

        # Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, return_attention=False):
        batch_size, seq_len = input_ids.size()

        # Create position indices
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        x = self.embedding(input_ids) + self.position_embedding(positions)

        # Store attention weights
        all_attentions = []

        # Pass through transformer layers
        for layer in self.layers:
            x, attn = layer(x)
            all_attentions.append(attn)

        # Final normalization
        x = self.layer_norm(x)

        # Use CLS token for classification
        cls_token = x[:, 0, :]
        logits = self.classifier(cls_token)

        if return_attention:
            return logits, all_attentions

        return logits
