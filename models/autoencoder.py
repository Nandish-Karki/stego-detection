''' 
import torch
import torch.nn as nn

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
        
            nn.Linear(16000, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 16000),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''
'''
import torch
import torch.nn as nn

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),   # [B, 16, 8000]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),  # [B, 32, 4000]
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),  # [B, 64, 2000]
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7), # [B, 128, 1000]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=15, stride=2, padding=7, output_padding=1),  # [B, 64, 2000]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=2, padding=7, output_padding=1),   # [B, 32, 4000]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=2, padding=7, output_padding=1),   # [B, 16, 8000]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=2, padding=7, output_padding=1),    # [B, 1, 16000]
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
'''

import torch
import torch.nn as nn

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=2, padding=4),     # [B, 8, 8000]
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=2, padding=4),    # [B, 16, 4000]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),   # [B, 32, 2000]
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, stride=2, padding=4),   # [B, 32, 1000]
            nn.ReLU(),
            nn.Dropout(0.2)  # Regularization to avoid overfitting
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=9, stride=2, padding=4, output_padding=1),   # [B, 32, 2000]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),   # [B, 16, 4000]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=9, stride=2, padding=4, output_padding=1),    # [B, 8, 8000]
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=9, stride=2, padding=4, output_padding=1),     # [B, 1, 16000]
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioAutoencoder(nn.Module):
    def __init__(self):
        super(AudioAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=2, padding=4),   # → L/2
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=9, stride=2, padding=4),  # → L/4
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4), # → L/8
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=9, stride=2, padding=4), # → L/16
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        original_len = x.shape[-1]

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Trim or pad output to match original length (in case of slight mismatch)
        if decoded.shape[-1] > original_len:
            decoded = decoded[:, :, :original_len]
        elif decoded.shape[-1] < original_len:
            pad_amt = original_len - decoded.shape[-1]
            decoded = F.pad(decoded, (0, pad_amt))

        return decoded
''' 