import torch
from torch.utils.data import DataLoader
from models.autoencoder import AudioAutoencoder
from data.loader import AudioDataset
import os
import numpy as np

def evaluate_autoencoder(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AudioDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = AudioAutoencoder().to(device)
    model.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location=device))
    model.eval()

    errors = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = ((outputs - inputs) ** 2).mean(dim=1)
            errors.extend(loss.cpu().numpy())

    print("Reconstruction error stats:")
    print("Mean:", np.mean(errors))
    print("Std:", np.std(errors))

if __name__ == '__main__':
    evaluate_autoencoder("data/test")
