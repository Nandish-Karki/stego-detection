'''
import torch
import torchaudio
import torch.nn.functional as F
from models.wavecnn import WaveCNN
from models.autoencoder import AudioAutoencoder


class AudioClassifier:
    def __init__(self, reconstruction_threshold=0.015):
        self.device = torch.device("cpu")
        self.reconstruction_threshold = reconstruction_threshold

        # Load WaveCNN
        self.wavecnn = WaveCNN()
        self.wavecnn.load_state_dict(torch.load("models/wavecnn_trained.pth", map_location=self.device))
        self.wavecnn.eval()

        # Load Autoencoder
        self.autoencoder = AudioAutoencoder()
        self.autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location=self.device))
        self.autoencoder.eval()

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)[:16000]
        if len(waveform) < 16000:
            waveform = F.pad(waveform, (0, 16000 - len(waveform)))
        input_tensor = waveform.unsqueeze(0)  # Shape: [1, 16000]
        
        return input_tensor

    def predict(self, audio_path, mode="cnn"):
        assert mode in ["cnn", "autoencoder", "hybrid"], "Mode must be 'cnn', 'autoencoder', or 'hybrid'"

        input_tensor = self.preprocess(audio_path)

        with torch.no_grad():
            # CNN prediction
            class_output = self.wavecnn(input_tensor)
            cnn_pred = torch.argmax(class_output, dim=1).item()

            # Autoencoder prediction
            reconstructed = self.autoencoder(input_tensor)
            reconstruction_error = F.mse_loss(reconstructed, input_tensor).item()

            # Decision logic
            if mode == "cnn":
                prediction = cnn_pred

            elif mode == "autoencoder":
                prediction = 0 if reconstruction_error < self.reconstruction_threshold else 1

            elif mode == "hybrid":
                if cnn_pred == 1 and reconstruction_error < self.reconstruction_threshold:
                    prediction = 0  # Override to Clean
                else:
                    prediction = cnn_pred

        return prediction, reconstruction_error

'''
''' 
import torch
import torchaudio
import torch.nn.functional as F
from models.wavecnn import WaveCNN
from models.autoencoder import AudioAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
import os


class AudioClassifier:
    def __init__(self, reconstruction_threshold=0.015):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_threshold = reconstruction_threshold

        # Load WaveCNN
        self.wavecnn = WaveCNN().to(self.device)
        self.wavecnn.load_state_dict(torch.load("models/wavecnn_trained.pth", map_location=self.device))
        self.wavecnn.eval()

        # Load Autoencoder
        self.autoencoder = AudioAutoencoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location=self.device))
        self.autoencoder.eval()

        # Load Denoising Autoencoder
        self.denoiser = DenoisingAutoencoder().to(self.device)
        self.denoiser.load_state_dict(torch.load("models/denoising_autoencoder.pth", map_location=self.device))
        self.denoiser.eval()

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        return waveform, sr

    def predict(self, audio_path, mode="cnn"):
        assert mode in ["cnn", "autoencoder", "hybrid"], "Mode must be 'cnn', 'autoencoder', or 'hybrid'"

        waveform, sr = self.preprocess(audio_path)
        input_tensor = waveform.unsqueeze(0).to(self.device)  # [1, T]

        with torch.no_grad():
            # CNN prediction
            cnn_input = input_tensor
            if cnn_input.shape[-1] < 64000:
                cnn_input = F.pad(cnn_input, (0, 64000 - cnn_input.shape[-1]))
            cnn_output = self.wavecnn(cnn_input)
            cnn_pred = torch.argmax(cnn_output, dim=1).item()

            # Autoencoder prediction
            ae_input = input_tensor[:, :16000]
            print(ae_input.shape)  #
            if ae_input.shape[-1] < 16000:
                ae_input = F.pad(ae_input, (0, 16000 - ae_input.shape[-1]))
            ae_recon = self.autoencoder(ae_input)
            reconstruction_error = F.mse_loss(ae_recon, ae_input).item()

            # Final prediction
            if mode == "cnn":
                prediction = cnn_pred
            elif mode == "autoencoder":
                prediction = 0 if reconstruction_error < self.reconstruction_threshold else 1
            elif mode == "hybrid":
                if cnn_pred == 1 and reconstruction_error < self.reconstruction_threshold:
                    prediction = 0
                else:
                    prediction = cnn_pred

        # Run denoising only if Stego
        denoised_path = None
        if prediction == 1:
            segment_length = 16000
            num_segments = input_tensor.shape[-1] // segment_length
            denoised_segments = []

            for i in range(num_segments):
                segment = input_tensor[:, i * segment_length : (i + 1) * segment_length]
                if segment.shape[-1] < segment_length:
                    segment = F.pad(segment, (0, segment_length - segment.shape[-1]))
                segment = segment.unsqueeze(1)  # [1, 1, T]
                denoised = self.denoiser(segment).squeeze(1)  # [1, T]
                denoised_segments.append(denoised)

            remainder = input_tensor.shape[-1] % segment_length
            if remainder > 0:
                tail = input_tensor[:, -remainder:]
                tail = F.pad(tail, (0, segment_length - remainder))
                tail = tail.unsqueeze(1)
                denoised_tail = self.denoiser(tail).squeeze(1)
                denoised_segments.append(denoised_tail[:, :remainder])

            full_denoised = torch.cat(denoised_segments, dim=1)  # [1, T]
            os.makedirs("outputs", exist_ok=True)
            denoised_path = "outputs/denoised_output.wav"
            torchaudio.save(denoised_path, full_denoised.detach().cpu(), sr)
            print(f"[INFO] Saved denoised audio of shape {full_denoised.shape} to {denoised_path} at {sr}Hz")

        return prediction, reconstruction_error, denoised_path
''' 
''' 
import torch
import torchaudio
import torch.nn.functional as F
from models.wavecnn import WaveCNN
from models.autoencoder import AudioAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
import os


class AudioClassifier:
    def __init__(self, reconstruction_threshold=0.125147):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_threshold = reconstruction_threshold

        # Load WaveCNN
        self.wavecnn = WaveCNN().to(self.device)
        self.wavecnn.load_state_dict(torch.load("models/wavecnn_trained.pth", map_location=self.device))
        self.wavecnn.eval()

        # Load Autoencoder
        self.autoencoder = AudioAutoencoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location=self.device))
        self.autoencoder.eval()

        # Load Denoising Autoencoder
        self.denoiser = DenoisingAutoencoder().to(self.device)
        self.denoiser.load_state_dict(torch.load("models/denoising_autoencoder.pth", map_location=self.device))
        self.denoiser.eval()

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        return waveform, sr

    def split_chunks(self, waveform, chunk_size):
        chunks = []
        for i in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[i:i + chunk_size]
            if chunk.shape[0] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))
                chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)

            chunks.append(chunk.unsqueeze(0))  # [1, T]
        return torch.stack(chunks).to(self.device)  # [N, 1, T]

    def predict(self, audio_path, mode="cnn"):
        assert mode in ["cnn", "autoencoder", "hybrid"], "Mode must be 'cnn', 'autoencoder', or 'hybrid'"

        waveform, sr = self.preprocess(audio_path)
        chunk_size = 16000
        chunks = self.split_chunks(waveform, chunk_size)  # [N, 1, 16000]

        with torch.no_grad():
            # CNN prediction (chunk-wise)
            cnn_preds = []
            for chunk in chunks:
                out = self.wavecnn(chunk)
                pred = torch.argmax(out, dim=1).item()
                cnn_preds.append(pred)
            cnn_majority = max(set(cnn_preds), key=cnn_preds.count)

            # Autoencoder prediction (chunk-wise reconstruction error)
            ae_errors = []
            for chunk in chunks:
                recon = self.autoencoder(chunk)
                err = F.mse_loss(recon, chunk).item()
                ae_errors.append(err)
            avg_error = sum(ae_errors) / len(ae_errors)

            # Final decision
            if mode == "cnn":
                prediction = cnn_majority
            elif mode == "autoencoder":
                prediction = 0 if avg_error < self.reconstruction_threshold else 1
            elif mode == "hybrid":
                if cnn_majority == 1 and avg_error < self.reconstruction_threshold:
                    prediction = 0
                else:
                    prediction = cnn_majority

        # Denoising step (if stego detected)
        denoised_path = None
        if prediction == 1:
            denoised_segments = []
            for chunk in chunks:
                denoised = self.denoiser(chunk).squeeze(1)  # [1, T]
                denoised_segments.append(denoised)
            full_denoised = torch.cat(denoised_segments, dim=1)  # [1, T]
            os.makedirs("outputs", exist_ok=True)
            denoised_path = "outputs/denoised_output.wav"
            torchaudio.save(denoised_path, full_denoised.detach().cpu(), sr)
            print(f"[INFO] Saved denoised audio of shape {full_denoised.shape} to {denoised_path} at {sr}Hz")

        return prediction, avg_error, denoised_path
'''
import torch
import torchaudio
import torch.nn.functional as F
from models.wavecnn import WaveCNN
from models.autoencoder import AudioAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
import os


class AudioClassifier:
    def __init__(self, reconstruction_threshold=0.125147):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_threshold = reconstruction_threshold

        # Load WaveCNN
        self.wavecnn = WaveCNN().to(self.device)
        self.wavecnn.load_state_dict(torch.load("models/wavecnn_trained.pth", map_location=self.device))
        self.wavecnn.eval()

        # Load Autoencoder
        self.autoencoder = AudioAutoencoder().to(self.device)
        self.autoencoder.load_state_dict(torch.load("models/autoencoder_trained.pth", map_location=self.device))
        self.autoencoder.eval()

        # Load Denoising Autoencoder
        self.denoiser = DenoisingAutoencoder().to(self.device)
        self.denoiser.load_state_dict(torch.load("models/denoising_autoencoder.pth", map_location=self.device))
        self.denoiser.eval()

    def preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        return waveform, sr

    def split_chunks(self, waveform, chunk_size):
        chunks = []
        for i in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[i:i + chunk_size]
            if chunk.shape[0] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))
            # ✅ Normalize the chunk (match training)
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)
            chunks.append(chunk.unsqueeze(0))  # [1, T]
        return torch.stack(chunks).to(self.device)  # [N, 1, T]

    def predict(self, audio_path, mode="cnn"):
        assert mode in ["cnn", "autoencoder", "hybrid"], "Mode must be 'cnn', 'autoencoder', or 'hybrid'"

        waveform, sr = self.preprocess(audio_path)
        chunk_size = 16000
        chunks = self.split_chunks(waveform, chunk_size)  # [N, 1, 16000]

        with torch.no_grad():
            # CNN prediction using softmax averaging
            all_probs = []
            for chunk in chunks:
                out = self.wavecnn(chunk)  # [1, 2]
                prob = F.softmax(out, dim=1)
                all_probs.append(prob.squeeze(0))  # [2]

            avg_prob = torch.stack(all_probs).mean(dim=0)  # [2]
            cnn_majority = torch.argmax(avg_prob).item()

            # Autoencoder reconstruction error
            ae_errors = []
            for chunk in chunks:
                recon = self.autoencoder(chunk)
                err = F.mse_loss(recon, chunk).item()
                ae_errors.append(err)
            avg_error = sum(ae_errors) / len(ae_errors)

            # Final prediction
            if mode == "cnn":
                prediction = cnn_majority
            elif mode == "autoencoder":
                prediction = 0 if avg_error < self.reconstruction_threshold else 1
            elif mode == "hybrid":
                if cnn_majority == 1 and avg_error < self.reconstruction_threshold:
                    prediction = 0
                else:
                    prediction = cnn_majority

        # Denoise if stego
        denoised_path = None
        if prediction == 1:
            denoised_segments = []
            for chunk in chunks:
                denoised = self.denoiser(chunk).squeeze(1)  # [1, T]
                denoised_segments.append(denoised)
            full_denoised = torch.cat(denoised_segments, dim=1)  # [1, T]
            os.makedirs("outputs", exist_ok=True)
            denoised_path = "outputs/denoised_output.wav"
            torchaudio.save(denoised_path, full_denoised.detach().cpu(), sr)
            #print(f"[INFO] Saved denoised audio to {denoised_path} ({full_denoised.shape}, {sr}Hz)")

        return prediction, avg_error, denoised_path
