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

import torch
import torchaudio
import torch.nn.functional as F
from models.wavecnn import WaveCNN
from models.autoencoder import AudioAutoencoder
from models.denoising_autoencoder import DenoisingAutoencoder
# from data.message import extract_message_from_diff,decode_base64_payload
import os
import re


class AudioClassifier:
    def __init__(self, reconstruction_threshold=0.125147):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reconstruction_threshold = reconstruction_threshold

        # Load WaveCNN
        self.wavecnn = WaveCNN().to(self.device)
        self.wavecnn.load_state_dict(torch.load("models/wavecnn_best.pth", map_location=self.device))
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
        waveform = waveform.mean(dim=0)
        return waveform, sr
    
    def denoise_preprocess(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0)  # Convert to mono
        if waveform.shape[-1] < 176400:
            waveform = F.pad(waveform, (0, 176400 - waveform.shape[-1]))
        else:
            waveform = waveform[:176400]
        return waveform.unsqueeze(0), sr
    # @staticmethod
    # def parse_embedding_percent(filename):
    #     match = re.search(r"(\d+)_LSB", filename)
    #     if match:
    #         return int(match.group(1))
    #     return 100  # Default fallback
    def split_chunks(self, waveform, chunk_size):
        chunks = []
        for i in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[i:i + chunk_size]
            if chunk.shape[0] < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - chunk.shape[0]))
            #  Normalize the chunk (match training)
            chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-5)
            chunks.append(chunk.unsqueeze(0))  # [1, T]
        return torch.stack(chunks).to(self.device)  # [N, 1, T]

    def predict(self, audio_path, mode="cnn"):
        assert mode in ["cnn", "autoencoder", "hybrid"], "Mode must be 'cnn', 'autoencoder', or 'hybrid'"


        

        with torch.no_grad():
            # CNN prediction (chunk-wise)
            #cnn_preds = []
            #for chunk in chunks:
            #    out = self.wavecnn(chunk)
            #    pred = torch.argmax(out, dim=1).item()
            #    cnn_preds.append(pred)
            #cnn_majority = max(set(cnn_preds), key=cnn_preds.count)

            # CNN prediction using softmax averaging
            # all_probs = []
            # for chunk in chunks:
            #     out = self.wavecnn(chunk)  # [1, 2]
            #     prob = F.softmax(out, dim=1)
            #     all_probs.append(prob.squeeze(0))  # [2]

            # avg_prob = torch.stack(all_probs).mean(dim=0)  # [2]
            # cnn_majority = torch.argmax(avg_prob).item()
            waveform, sr = self.denoise_preprocess(audio_path)
            waveform = waveform.to(self.device)

            norm_waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
            out = self.wavecnn(norm_waveform)  # Shape: [1, 5] for 5-class classification
            prob = F.softmax(out, dim=1)  # [1, 5]
            print(f"[DEBUG] CNN Output Probabilities: {prob.squeeze().tolist()}")
            cnn_majority = torch.argmax(prob, dim=1).item()

            # # Autoencoder reconstruction error
            waveform, sr = self.preprocess(audio_path)
            chunk_size = 176400
            chunks = self.split_chunks(waveform, chunk_size)  # [N, 1, 16000]
            ae_errors = []
            for chunk in chunks:
                recon = self.autoencoder(chunk)
                err = F.mse_loss(recon, chunk).item()
                ae_errors.append(err)
                
            avg_error = sum(ae_errors) / len(ae_errors)
            print(f"[DEBUG] Average reconstruction error: {avg_error:.6f}")

            # Final prediction
            if mode == "cnn":
                prediction = cnn_majority
            elif mode == "autoencoder":
                prediction = 0 if avg_error < self.reconstruction_threshold else 1
            elif mode == "hybrid":
                if cnn_majority in(1,2,3,4) and avg_error < self.reconstruction_threshold:
                    prediction = 0
                else:
                    prediction = cnn_majority
        
        # Denoise if stego
        denoised_path = None
        diff_path = None
        message_path = None
        decoded_payload_path = None
        if prediction in (1,2,3,4):  # Assuming 1 is stego class
            waveform, sr = self.denoise_preprocess(audio_path)
            waveform = waveform.to(self.device)
            norm_waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)
            denoised = self.denoiser(norm_waveform)
            os.makedirs("outputs", exist_ok=True)
            denoised_path = "outputs/denoised_output.wav"
            torchaudio.save(denoised_path, denoised.detach().cpu(), sr)

            # Save difference for message retrieval
            diff = waveform.cpu() - denoised.cpu()
            diff_path = "outputs/difference.wav"
            torchaudio.save(diff_path, diff.detach(), sr)
            # embedding_percent = self.parse_embedding_percent(audio_path)
            # _, message = extract_message_from_diff(diff, embedding_percent=embedding_percent)
            # message_path = "outputs/extracted_message.txt"
            # with open(message_path, "w",encoding='utf-8') as f:
            #     f.write(message)
            
            # # --- New automatic Base64 decode and extraction ---
            # # Detect Base64 encoded message (heuristic: presence of BASE64 chars)
            # base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
            # if all(c in base64_chars for c in message.strip()):
            #     print("[INFO] Detected Base64 encoded message, decoding...")
            #     decoded_payload_path = decode_base64_payload(message, output_dir="outputs")
            # else:
        
            #     print("[INFO] Extracted message not detected as Base64 encoded.")
        return prediction,avg_error, denoised_path , diff_path #, message_path,decoded_payload_path

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
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-5)  # Normalize
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

        waveform, sr = self.preprocess(audio_path)  # [T]
        waveform_tensor = waveform.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T]

        if mode == "autoencoder":
            with torch.no_grad():
                recon = self.autoencoder(waveform_tensor)
                ae_error = F.mse_loss(recon, waveform_tensor).item()
                print(f"[DEBUG] AE Reconstruction Error: {ae_error:.6f}")
                prediction = 0 if ae_error < self.reconstruction_threshold else 1

                denoised_path = None
                if prediction == 1:
                    denoised = self.denoiser(waveform_tensor).squeeze(0)  # [1, T]
                    os.makedirs("outputs", exist_ok=True)
                    denoised_path = "outputs/denoised_output.wav"
                    torchaudio.save(denoised_path, denoised.detach().cpu(), sr)
                return prediction, ae_error, denoised_path

        # If CNN or Hybrid mode → use chunking
        chunk_size = 16000
        chunks = self.split_chunks(waveform, chunk_size)  # [N, 1, 16000]

        with torch.no_grad():
            all_probs = []
            for chunk in chunks:
                out = self.wavecnn(chunk)
                prob = F.softmax(out, dim=1)
                all_probs.append(prob.squeeze(0))  # [2]
            avg_prob = torch.stack(all_probs).mean(dim=0)
            cnn_majority = torch.argmax(avg_prob).item()

            # Autoencoder score (used in hybrid)
            ae_error = None
            if mode == "hybrid":
                recon = self.autoencoder(waveform_tensor)
                ae_error = F.mse_loss(recon, waveform_tensor).item()
                print(f"[DEBUG] AE Reconstruction Error: {ae_error:.6f}")

                if cnn_majority == 1 and ae_error < self.reconstruction_threshold:
                    prediction = 0
                else:
                    prediction = cnn_majority
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

            return prediction, ae_error, denoised_path
'''
