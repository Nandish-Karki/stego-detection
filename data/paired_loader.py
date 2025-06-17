import os
import torchaudio
from torch.utils.data import Dataset

class PairedAudioDataset(Dataset):
    def __init__(self, cover_dir, stego_dir):
        self.pairs = []
        for cover_file in os.listdir(cover_dir):
            if not cover_file.endswith(".wav"):
                continue
            cover_path = os.path.join(cover_dir, cover_file)
            base_name = cover_file.replace(".wav", "")
            for p in ["25perc", "50perc", "75perc", "100perc"]:
                # Match stego format: {base}_{base}.wav.{percent}_LSBR.wav
                stego_filename = f"{base_name}_{base_name}.wav.{p}_LSBR.wav"
                stego_path = os.path.join(stego_dir, stego_filename)
                if os.path.exists(stego_path):
                    self.pairs.append((stego_path, cover_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        stego_path, cover_path = self.pairs[idx]
        stego_waveform, _ = torchaudio.load(stego_path)
        cover_waveform, _ = torchaudio.load(cover_path)
        # Convert to mono and crop/pad to 16000 samples
        stego = stego_waveform.mean(dim=0)[:16000]
        cover = cover_waveform.mean(dim=0)[:16000]
        if len(stego) < 16000:
            stego = torch.nn.functional.pad(stego, (0, 16000 - len(stego)))
            cover = torch.nn.functional.pad(cover, (0, 16000 - len(cover)))
        return stego.unsqueeze(0), cover.unsqueeze(0)  # [1, T] each
