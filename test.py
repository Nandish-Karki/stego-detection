'''
import torch
import torchaudio
import torch.nn.functional as F
from models.autoencoder import AudioAutoencoder
import os

audio_folder = "audio/cover/"
errors = []

# Load model once
autoencoder = AudioAutoencoder()
autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location="cpu"))
autoencoder.eval()

# Process audio files
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_folder, filename)
        waveform, sample_rate = torchaudio.load(file_path)

        # Preprocess waveform
        waveform = waveform.mean(dim=0)[:64000]
        if len(waveform) < 64000:
            waveform = F.pad(waveform, (0, 64000 - len(waveform)))
        
        input_tensor = waveform.unsqueeze(0)  # [1, T]

        # Compute reconstruction error
        with torch.no_grad():
            reconstructed = autoencoder(input_tensor)
            error = F.mse_loss(reconstructed, input_tensor).item()
            errors.append(error)

# Print the mean reconstruction error
if errors:
    print("Mean reconstruction error:", sum(errors) / len(errors))
else:
    print("No audio files found.")

'''
'''
import torch
from models.denoising_autoencoder import DenoisingAutoencoder

x = torch.randn(1, 1, 16000)  # Simulate 1 second audio
model = DenoisingAutoencoder()
y = model(x)
print(f"Input shape: {x.shape}, Output shape: {y.shape}")
'''
''' 
import torch
import torchaudio
import torch.nn.functional as F
from models.autoencoder import AudioAutoencoder
import os

audio_folder = "audio/cover/"
errors = []

# Load model once
autoencoder = AudioAutoencoder()
autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location="cpu"))
autoencoder.eval()

chunk_size = 16000  # 1 second at 16kHz
full_length = 64000  # 4 seconds

# Process audio files
for filename in os.listdir(audio_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(audio_folder, filename)
        waveform, sample_rate = torchaudio.load(file_path)

        # Ensure mono and fixed length 64000 samples
        waveform = waveform.mean(dim=0)[:full_length]
        if len(waveform) < full_length:
            waveform = F.pad(waveform, (0, full_length - len(waveform)))

        total_error = 0

        with torch.no_grad():
            # Process in 1-second chunks (16000 samples)
            for i in range(0, full_length, chunk_size):
                chunk = waveform[i:i + chunk_size].unsqueeze(0)  # [1, 16000]

                reconstructed = autoencoder(chunk)

                # Crop to shortest length (safety)
                min_len = min(reconstructed.shape[1], chunk.shape[1])
                reconstructed = reconstructed[:, :min_len]
                chunk = chunk[:, :min_len]

                chunk_error = F.mse_loss(reconstructed, chunk).item()
                total_error += chunk_error

            avg_error = total_error / (full_length // chunk_size)
            errors.append(avg_error)

if errors:
    mean_error = sum(errors) / len(errors)
    std_error = (sum((e - mean_error) ** 2 for e in errors) / len(errors)) ** 0.5

    k = 3  # Sensitivity multiplier

    threshold = mean_error + k * std_error

    print(f"Mean reconstruction error: {mean_error:.6f}")
    print(f"Standard deviation: {std_error:.6f}")
    print(f"Suggested threshold (mean + {k}*std): {threshold:.6f}")
else:
    print("No audio files found.")

'''
import os
import csv
from classifier import AudioClassifier

# Load classifier
classifier = AudioClassifier()

# Paths
labels_csv = "data/labels.csv"
audio_root = "audio"  # base directory for audio (should have /cover and /stego subdirs)
output_csv = "results/predictions.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Select number of test files
num_tests = 3171  # Set to an integer (e.g., 10) to limit; or None for all

# Read labels and run predictions
results = []
tested = 0
correct = 0

with open(labels_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        rel_path, true_label = row
        audio_path = os.path.join(audio_root, rel_path)
        true_label = int(true_label)

        pred, error, _ = classifier.predict(audio_path, mode="cnn")  # or "hybrid"
        results.append([rel_path, true_label, pred, round(error, 6)])

        if pred == true_label:
            correct += 1
        tested += 1

        if num_tests and tested >= num_tests:
            break

# Write to CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["File", "TrueLabel", "PredictedLabel", "AE_Error"])
    writer.writerows(results)

print(f"[✅] Saved {tested} predictions to {output_csv}")
print(f"[🔍] Accuracy: {correct}/{tested} = {correct/tested:.2%}")

