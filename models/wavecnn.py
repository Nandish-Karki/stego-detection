 
import torch
import torch.nn as nn

class WaveCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(WaveCNN, self).__init__()
        self.conv_layers = nn.Sequential(# Convolutional layers for feature extraction
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),#training stability and faster convergence
            nn.ReLU(), #non linearity
            nn.MaxPool1d(4), #downsampling the signal by 4 to focus on key features

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Adapts to input length Benefit: You can feed in variable-length audio!

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        
        x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv_layers(x)  # [B, C, T']
        x = self.global_pool(x)  # [B, C, 1]
        x = self.fc(x)  # [B, num_classes]
        return x
'''

class WaveCNN(nn.Module):
    def __init__(self):
        super(WaveCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output single logit
        )

    def forward(self, x):
        #x = x.unsqueeze(1)  # [B, 1, T]
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.fc(x).squeeze(1)  # [B]
        return x
    
''' 