# src/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os

from dataset import LogicDataset, build_vocab
from model import LogicTransformer

# ==========================================
# CONFIG
# ==========================================

BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_LEN = 64
MODEL_NAME = "best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# PATH HANDLING
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, MODEL_NAME)

TRAIN_PATH = os.path.join(DATA_DIR, "train.json")
TEST_PATH = os.path.join(DATA_DIR, "test.json")


# ==========================================
# TRAINING FUNCTION
# ==========================================

def train():
    print(f"Using device: {DEVICE}")
    print(f"Train file: {TRAIN_PATH}")
    print(f"Test file: {TEST_PATH}")

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"train.json not found at {TRAIN_PATH}. Run data_generator.py first.")

    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"test.json not found at {TEST_PATH}. Run data_generator.py first.")

    vocab = build_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    train_dataset = LogicDataset(TRAIN_PATH, vocab, MAX_LEN)
    test_dataset = LogicDataset(TEST_PATH, vocab, MAX_LEN)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=0)

    model = LogicTransformer(
        vocab_size=len(vocab),
        d_model=256,
        n_heads=8,
        num_layers=6,
        max_len=MAX_LEN
    ).to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    criterion = nn.CrossEntropyLoss()

    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # =====================
        # TRAIN
        # =====================
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # =====================
        # EVALUATION
        # =====================
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}\n")

        # Save best model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved! (Accuracy: {accuracy:.4f})")

    print("\n" + "="*50)
    print("Training complete.")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Model saved at: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
