# src/visualization.py

import torch


def extract_cls_attention(attentions):
    """
    Takes list of attention tensors from all layers.
    Returns average CLS attention over heads and layers.
    
    Args:
        attentions: list of (batch, heads, seq, seq) tensors
    
    Returns:
        Average attention from CLS token: (batch, seq)
    """
    cls_attentions = []

    for layer_attn in attentions:
        # Take CLS token attention (index 0)
        # layer_attn shape: (batch, heads, seq, seq)
        cls_layer = layer_attn[:, :, 0, :]  # (batch, heads, seq)
        cls_attentions.append(cls_layer)

    # Stack: (layers, batch, heads, seq)
    stacked = torch.stack(cls_attentions)

    # Average over layers and heads
    avg = stacked.mean(dim=(0, 2))  # (batch, seq)

    return avg


def rank_premises(tokens, cls_attention_scores):
    """
    Aggregates attention scores per premise.
    
    Args:
        tokens: list of token strings
        cls_attention_scores: attention scores for each position
    
    Returns:
        List of (premise_string, score) sorted by score descending
    """
    premise_scores = {}
    current_premise = []
    score_sum = 0.0

    for i, (token, score) in enumerate(zip(tokens, cls_attention_scores)):
        if token == ";":
            if current_premise:
                premise_str = " ".join(current_premise)
                premise_scores[premise_str] = score_sum
            current_premise = []
            score_sum = 0.0
        elif token == "[SEP]":
            if current_premise:
                premise_str = " ".join(current_premise)
                premise_scores[premise_str] = score_sum
            current_premise = []
            score_sum = 0.0
        elif token not in ["[CLS]", "[PAD]"]:
            current_premise.append(token)
            score_sum += float(score)

    # Handle remaining tokens (query part)
    if current_premise:
        premise_str = " ".join(current_premise)
        premise_scores[premise_str] = score_sum

    # Sort by score descending
    ranked = sorted(premise_scores.items(), key=lambda x: x[1], reverse=True)

    return ranked
