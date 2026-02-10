import os
import torch
from funasr import AutoModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataset import EmotionDataset
from tqdm import tqdm
from huggingface_hub import login
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Authentication ---
HF_TOKEN = "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"
login(token=HF_TOKEN)

# --- 2. Configuration ---
device_str = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5
EPOCHS = 10
BATCH_SIZE = 2
METADATA_CSV = "metadata.csv"

print(f"Using device: {device_str}")

# --- 3. Load Model ---
print("Loading emotion2vec_plus_base model...")
model_wrapper = AutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)
model = model_wrapper.model
model.to(device_str)

# --- 4. Freeze backbone, unfreeze last 2 blocks + projection head ---
print("Configuring trainable parameters...")
# First freeze everything
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 2 transformer blocks
if hasattr(model, 'blocks'):
    for block in model.blocks[-2:]:
        for param in block.parameters():
            param.requires_grad = True
    print(f"Unfroze last 2 blocks (out of {len(model.blocks)} total)")

# Unfreeze projection head
if hasattr(model, 'proj'):
    for param in model.proj.parameters():
        param.requires_grad = True
    print("Unfroze projection head")

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")


# --- 5. Data Preparation ---
def collate_fn(batch):
    """Collate function to pad variable-length audio sequences."""
    audios = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return audios_padded, labels


print("Loading datasets...")
train_ds = EmotionDataset(METADATA_CSV, split='train')
test_ds = EmotionDataset(METADATA_CSV, split='test')

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

# --- 6. Training Setup ---
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


def extract_logits(model, wavs):
    """
    Extract emotion logits from the emotion2vec model.
    Handles the dictionary output from extract_features.
    """
    # Get features from the backbone
    feat_output = model.extract_features(wavs)

    # Handle dictionary output
    if isinstance(feat_output, dict):
        # Try common keys: 'x', 'last_hidden_state', or use first value
        if 'x' in feat_output:
            features = feat_output['x']
        elif 'last_hidden_state' in feat_output:
            features = feat_output['last_hidden_state']
        else:
            features = next(iter(feat_output.values()))
    else:
        features = feat_output

    # Pool temporal dimension: [Batch, Time, Hidden] -> [Batch, Hidden]
    pooled_features = features.mean(dim=1)

    # Project to emotion classes
    logits = model.proj(pooled_features)

    return logits


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="Training")
    for wavs, labels in pbar:
        wavs, labels = wavs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = extract_logits(model, wavs)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for wavs, labels in pbar:
            wavs, labels = wavs.to(device), labels.to(device)

            # Forward pass
            logits = extract_logits(model, wavs)
            loss = criterion(logits, labels)

            # Track metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


# --- 7. Training Loop ---
print("\n" + "=" * 50)
print("Starting Training")
print("=" * 50)

best_test_acc = 0
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device_str)
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # Evaluate
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device_str)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save best model
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), "emotion2vec_finetuned_best.pth")
        print(f"✓ New best model saved! (Accuracy: {best_test_acc:.4f})")

# --- 8. Final Evaluation ---
print("\n" + "=" * 50)
print("Training Complete!")
print("=" * 50)
print(f"Best Test Accuracy: {best_test_acc:.4f}")

# Load best model and get detailed metrics
model.load_state_dict(torch.load("emotion2vec_finetuned_best.pth"))
_, _, final_preds, final_labels = evaluate(model, test_loader, criterion, device_str)

# Class names for report
class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised', 'unknown']
print("\nClassification Report:")
print(classification_report(final_labels, final_preds, target_names=class_names, zero_division=0))

# Save final model
torch.save(model.state_dict(), "emotion2vec_finetuned_final.pth")
print("\nFinal model saved as 'emotion2vec_finetuned_final.pth'")
print("Best model saved as 'emotion2vec_finetuned_best.pth'")