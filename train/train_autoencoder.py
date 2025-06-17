''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder

# Load cover only
dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv")
cover_dataset = [(x, y) for x, y in dataset if y == 0]
cover_loader = DataLoader(cover_dataset, batch_size=16, shuffle=True)

model = AudioAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for inputs, _ in cover_loader:
        inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.loader import AudioDataset
from models.autoencoder import AudioAutoencoder

# Only use cover files (label = 0)
dataset = AudioDataset(audio_dir="audio", labels_file="data/labels.csv", sample_len=16000)
cover_chunks = [x for x in dataset if x[1] == 0]
cover_loader = DataLoader(cover_chunks, batch_size=16, shuffle=True)

model = AudioAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for inputs, _ in cover_loader:
        inputs = inputs.unsqueeze(1)  # [B, 1, 16000]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/autoencoder_trained.pth")
print("✅ Autoencoder model saved to models/autoencoder_trained.pth")
