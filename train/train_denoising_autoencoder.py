import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.denoising_autoencoder import DenoisingAutoencoder
from data.paired_loader import PairedAudioDataset
import os

def train_denoising_autoencoder(cover_dir, stego_dir, epochs=20, batch_size=16, lr=1e-4):
    dataset = PairedAudioDataset(cover_dir, stego_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DenoisingAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for stego, cover in loader:
            output = model(stego)
            loss = criterion(output, cover)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/denoising_autoencoder.pth")
    print("✅ Saved to models/denoising_autoencoder.pth")

if __name__ == '__main__':
    train_denoising_autoencoder("audio/cover", "audio/stego")
