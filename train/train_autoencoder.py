''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder

# Load cover only
dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv" ,sample_len=16000)
cover_chunks = [x for x in dataset if x[1] == 0]
cover_loader = DataLoader(cover_chunks, batch_size=16, shuffle=True)
#cover_dataset = [(x, y) for x, y in dataset if y == 0]
#cover_loader = DataLoader(cover_dataset, batch_size=16, shuffle=True)

model = AudioAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for inputs, _ in cover_loader:
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/autoencoder_trained.pth")
print("Autoencoder model saved.")

'''
''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder
import os

# Config
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
SAMPLE_LEN = 16000
VAL_SPLIT = 0.1
MODEL_SAVE_PATH = "models/autoencoder_trained.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load dataset (only cover files: label == 0)
full_dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv", sample_len=SAMPLE_LEN)
cover_chunks = [x for x in full_dataset if x[1] == 0]

# Split into train/val
val_size = int(len(cover_chunks) * VAL_SPLIT)
train_size = len(cover_chunks) - val_size
train_dataset, val_dataset = random_split(cover_chunks, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = AudioAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, _ in train_loader:
        inputs = inputs.unsqueeze(1).to(device)  # [B, 1, 16000]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.unsqueeze(1).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" Saved improved model (val loss: {best_val_loss:.6f}) → {MODEL_SAVE_PATH}")
print("Training complete. Best model saved.")
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder
import os
import torch
print(torch.__version__)

# Config
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
SAMPLE_LEN = 176400
VAL_SPLIT = 0.1
MODEL_SAVE_PATH = "models/autoencoder_trained.pth"
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load dataset (only cover files: label == 0)
full_dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv", sample_len=SAMPLE_LEN)
cover_chunks = [x for x in full_dataset if x[1] == 0]

# Split into train/val
val_size = int(len(cover_chunks) * VAL_SPLIT)
train_size = len(cover_chunks) - val_size
train_dataset, val_dataset = random_split(cover_chunks, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model and training setup
model = AudioAutoencoder().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


best_val_loss = float("inf")
early_stop_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, _ in train_loader:
        inputs = inputs.unsqueeze(1).to(device)  # [B, 1, 16000]
        inputs = inputs.clamp(-1.0, 1.0)

        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.unsqueeze(1).to(device)
            inputs = inputs.clamp(-1.0, 1.0)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✔ Saved improved model (val loss: {best_val_loss:.6f}) → {MODEL_SAVE_PATH}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f" Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs).")
            break

print("Training complete. Best model saved.")

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder
import os

print(torch.__version__)

# Config
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.1
MODEL_SAVE_PATH = "models/autoencoder_trained.pth"
PATIENCE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load full dataset (no chunking)
full_dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv")

# Filter only cover files (label == 0)
cover_dataset = [item for item in full_dataset if item[1] == 0]

# Split into train and val sets
val_size = int(len(cover_dataset) * VAL_SPLIT)
train_size = len(cover_dataset) - val_size
train_dataset, val_dataset = random_split(cover_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_batch(x, device))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_batch(x, device))

# Collate function for dynamic padding
def collate_batch(batch, device):
    waveforms, _ = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in waveforms]
    padded_tensor = torch.stack(padded).unsqueeze(1).to(device)
    return padded_tensor, padded_tensor  # input == target for AE

# Model and training setup
model = AudioAutoencoder().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

best_val_loss = float("inf")
early_stop_counter = 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.clamp(-1.0, 1.0)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.clamp(-1.0, 1.0)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  ✔ Saved improved model (val loss: {best_val_loss:.6f}) → {MODEL_SAVE_PATH}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print(f" Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs).")
            break

print("Training complete. Best model saved.")
''' 

