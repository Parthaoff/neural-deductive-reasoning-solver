# src/evaluate.py

import torch
import torch.nn.functional as F
import os
import json

from dataset import build_vocab, LogicDataset, logic_tokenizer, LABEL_MAP_INV
from model import LogicTransformer
from visualization import extract_cls_attention, rank_premises


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "best_model.pt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "test.json")

MAX_LEN = 64


def load_model():
    """Load the trained model."""
    vocab = build_vocab()
    
    model = LogicTransformer(
        vocab_size=len(vocab),
        max_len=MAX_LEN
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: No model found at {MODEL_PATH}")
    
    model.eval()
    return model, vocab


def evaluate_sample(sample_index=0):
    """Evaluate a single sample from the test set."""
    vocab = build_vocab()
    dataset = LogicDataset(DATA_PATH, vocab, MAX_LEN)
    model, _ = load_model()

    sample = dataset.data[sample_index]
    input_ids, label = dataset[sample_index]
    input_ids = input_ids.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attentions = model(input_ids, return_attention=True)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    cls_attention = extract_cls_attention(attentions)[0].cpu()

    tokens = dataset.tokenize(sample["premises"], sample["query"])

    print("\n" + "="*50)
    print("SAMPLE EVALUATION")
    print("="*50)
    print(f"\nPremises: {sample['premises']}")
    print(f"Query: {sample['query']}")
    print(f"True Label: {sample['label']}")
    print(f"\nTokens: {tokens}")
    print(f"\nPrediction: {LABEL_MAP_INV[prediction]}")
    print(f"Confidence: {probs[0][prediction].item():.4f}")
    print(f"Correct: {LABEL_MAP_INV[prediction] == sample['label']}")

    ranked = rank_premises(tokens, cls_attention)

    print("\nRanked Premises (Proof Path):")
    for i, (premise, score) in enumerate(ranked):
        print(f"  Step {i+1}: {premise}  ->  {score:.4f}")


def evaluate_custom(premises, query):
    """Evaluate a custom logic problem."""
    model, vocab = load_model()

    # Tokenize
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

    # Encode
    ids = [vocab[t] for t in tokens if t in vocab]
    if len(ids) < MAX_LEN:
        ids += [vocab["[PAD]"]] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    input_ids = torch.tensor(ids).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attentions = model(input_ids, return_attention=True)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    cls_attention = extract_cls_attention(attentions)[0].cpu()
    ranked = rank_premises(tokens, cls_attention)

    print("\n" + "="*50)
    print("CUSTOM EVALUATION")
    print("="*50)
    print(f"\nPremises: {premises}")
    print(f"Query: {query}")
    print(f"\nPrediction: {LABEL_MAP_INV[prediction]}")
    print(f"Confidence: {probs[0][prediction].item():.4f}")

    print("\nReasoning Steps:")
    for i, (premise, score) in enumerate(ranked):
        print(f"  Step {i+1}: {premise}  (importance: {score:.4f})")

    return LABEL_MAP_INV[prediction], probs[0][prediction].item(), ranked


if __name__ == "__main__":
    print("Evaluating sample from test set...")
    evaluate_sample(0)
    
    print("\n\n" + "="*70 + "\n")
    
    print("Evaluating custom problem...")
    evaluate_custom(
        premises=["A -> B", "B -> C", "A"],
        query="C"
    )
