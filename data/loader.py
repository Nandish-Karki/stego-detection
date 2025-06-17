'''
import torch
from torch.utils.data import Dataset
import torchaudio
import os

class AudioDataset(Dataset):
    def __init__(self, audio_dir, labels_file=None, transform=None, sample_len=16000):
        self.audio_dir = audio_dir
        self.transform = transform
        self.sample_len = sample_len
        self.labels = {}

        if labels_file:
            with open(labels_file, 'r') as f:
                for line in f:
                    path, label = line.strip().split(',')
                    self.labels[path] = int(label)

        self.audio_files = list(self.labels.keys()) if labels_file else os.listdir(audio_dir)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_name = self.audio_files[idx]
        file_path = os.path.join(self.audio_dir, file_name)
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)  # Convert to mono

        # Pad or truncate to sample_len (e.g., 1s of 16kHz)
        if len(waveform) < self.sample_len:
            waveform = torch.nn.functional.pad(waveform, (0, self.sample_len - len(waveform)))
        else:
            waveform = waveform[:self.sample_len]

        # Normalize waveform
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)

        if self.transform:
            waveform = self.transform(waveform)

        label = self.labels[file_name] if self.labels else -1
        return waveform, label

'''
import torch
from torch.utils.data import Dataset
import torchaudio
import os

class AudioDataset(Dataset):
    def __init__(self, audio_dir, labels_file=None, transform=None, sample_len=16000):
        self.audio_dir = audio_dir
        self.transform = transform
        self.sample_len = sample_len
        self.labels = {}
        self.audio_segments = []

        if labels_file:
            with open(labels_file, 'r') as f:
                for line in f:
                    path, label = line.strip().split(',')
                    full_path = os.path.join(audio_dir, path)
                    if os.path.exists(full_path):
                        waveform, sr = torchaudio.load(full_path)
                        waveform = waveform.mean(dim=0)  # mono
                        num_segments = len(waveform) // sample_len
                        for i in range(num_segments):
                            self.audio_segments.append((full_path, i, int(label)))
        else:
            for fname in os.listdir(audio_dir):
                if fname.endswith(".wav"):
                    full_path = os.path.join(audio_dir, fname)
                    waveform, sr = torchaudio.load(full_path)
                    waveform = waveform.mean(dim=0)  # mono
                    num_segments = len(waveform) // sample_len
                    for i in range(num_segments):
                        self.audio_segments.append((full_path, i, -1))

    def __len__(self):
        return len(self.audio_segments)

    def __getitem__(self, idx):
        file_path, chunk_idx, label = self.audio_segments[idx]
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform.mean(dim=0)

        start = chunk_idx * self.sample_len
        end = start + self.sample_len
        chunk = waveform[start:end]

        if len(chunk) < self.sample_len:
            chunk = torch.nn.functional.pad(chunk, (0, self.sample_len - len(chunk)))

        chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)  # Normalize

        if self.transform:
            chunk = self.transform(chunk)

        return chunk, label


