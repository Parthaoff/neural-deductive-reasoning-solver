# src/dataset.py

import json
import torch
import string
from torch.utils.data import Dataset

# ==========================================================
# VOCABULARY
# ==========================================================

VARIABLES = list(string.ascii_uppercase)
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]"]
OPERATORS = ["->", "~", ";"]

VOCAB_LIST = SPECIAL_TOKENS + OPERATORS + VARIABLES

LABEL_MAP = {
    "True": 0,
    "False": 1,
    "Unknown": 2
}

LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def build_vocab():
    return {token: idx for idx, token in enumerate(VOCAB_LIST)}


def get_inv_vocab():
    vocab = build_vocab()
    return {idx: token for token, idx in vocab.items()}


# ==========================================================
# TOKENIZER
# ==========================================================

def logic_tokenizer(expression):
    """
    Properly tokenizes logical expressions.
    Handles: A, ~A, A -> B
    """
    tokens = []
    i = 0
    expression = expression.strip()

    while i < len(expression):
        # Check for implication
        if expression[i:i+2] == "->":
            tokens.append("->")
            i += 2
        # Check for negation
        elif expression[i] == "~":
            tokens.append("~")
            i += 1
        # Check for variable (uppercase letter)
        elif expression[i].isupper() and expression[i] in VARIABLES:
            tokens.append(expression[i])
            i += 1
        # Check for semicolon
        elif expression[i] == ";":
            tokens.append(";")
            i += 1
        # Skip spaces and unknown characters
        elif expression[i] == " ":
            i += 1
        else:
            # Skip unknown characters silently
            i += 1

    return tokens


# ==========================================================
# DATASET CLASS
# ==========================================================

class LogicDataset(Dataset):

    def __init__(self, json_path, vocab, max_len=64):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.max_len = max_len

    def tokenize(self, premises, query):
        tokens = ["[CLS]"]

        for p in premises:
            p_tokens = logic_tokenizer(p)
            if p_tokens:
                tokens.extend(p_tokens)
                tokens.append(";")

        tokens.append("[SEP]")
        
        q_tokens = logic_tokenizer(query)
        if q_tokens:
            tokens.extend(q_tokens)

        return tokens

    def encode(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            # Skip unknown tokens

        # Pad or truncate
        if len(ids) < self.max_len:
            ids += [self.vocab["[PAD]"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]

        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        tokens = self.tokenize(sample["premises"], sample["query"])
        input_ids = self.encode(tokens)

        label = torch.tensor(LABEL_MAP[sample["label"]], dtype=torch.long)

        return input_ids, label
