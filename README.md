# Audio Steganalysis using Deep Learning

This project focuses on detecting and removing hidden messages embedded in audio using steganographic techniques like LSB (Least Significant Bit) encoding. The pipeline is designed using PyTorch, CNN architectures, Autoencoders, and provides an interactive interface via Gradio.

---

## 🚀 Features

- ✅ Detection of steganographic content using **WaveCNN**
- ✅ Denoising/removal of hidden messages using **Autoencoder**
- ✅ Support for multiple stego levels (25%, 50%, 75%, 100%)
- ✅ Trained models (.pth) for fast inference
- ✅ Evaluation scripts and metrics
- ✅ Gradio-based demo interface
- ✅ Dockerfile for containerized deployment

---

## 🗂 Project Structure

IT_SECURITY/
├── audio/ # Input audio files (cover, stego, text) (create it on your own with the data with 4sec audio)
├── data/ # Optional data folder
├── evaluate/ # Evaluation scripts
├── models/ # Trained models and model architectures
├── outputs/ # Denoised audio outputs
├── results/ # Inference results (e.g., predictions.csv)
├── train/ # Training scripts
├── venv/ # Local virtual environment (excluded)
├── Dockerfile # Docker container setup
├── run.py # Gradio UI runner
├── requirement.txt # Python dependencies
├── .gitignore # Git ignore rules
├── .dockerignore # Docker ignore rules
├── README.md # This file


---

## 🧪 Setup & Usage

### 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Audio-Steganalysis.git
cd Audio-Steganalysis

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirement.txt

▶️ Run Detection
python run.py


🏋️‍♀️ Training Models
Train Autoencoder:
python train/train_autoencoder.py

Train WaveCNN:
python train/train_wavecnn.py

Train Denoising Autoencoder:
python train/train_denoising_autoencoder.py


🐳 Docker
Build and run using Docker:

docker build -t audio-steganalysis .
docker run -p 7860:7860 audio-steganalysis

For questions or collaboration, feel free to reach out at:
nkarki2791@gmail.com




