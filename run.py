'''
import argparse
import gradio as gr
from classifier import AudioClassifier
import os

# Initialize classifier
classifier = AudioClassifier()

# Shared prediction function
def detect_stego(audio_path, mode):
    predicted_class, reconstruction_error = classifier.predict(audio_path, mode=mode)
    class_label = "Stego" if predicted_class == 1 else "Cover"
    return f"Mode: {mode.upper()}\nPrediction: {class_label}\nReconstruction Error: {reconstruction_error:.6f}"

def main():
    parser = argparse.ArgumentParser(description="Audio Steganography Detection")
    parser.add_argument("--file", type=str, help="Path to input WAV file")
    parser.add_argument("--mode", type=str, choices=["cnn", "autoencoder", "hybrid"], default="hybrid", help="Detection mode")
    parser.add_argument("--nogui", action="store_true", help="Run without launching the Gradio web app")

    args = parser.parse_args()

    if args.nogui:
        if not args.file or not os.path.isfile(args.file):
            print("❌ Please provide a valid file path using --file when --nogui is set.")
            return
        result = detect_stego(args.file, args.mode)
        print("✅ Classification result:\n", result)

        # Optional: write to output.txt
        with open("output.txt", "w") as f:
            f.write(result + "\n")
    else:
        # Run Gradio GUI
        gui = gr.Interface(
            fn=detect_stego,
            inputs=[
                gr.Audio(type="filepath", label="Upload WAV Audio"),
                gr.Radio(["cnn", "autoencoder", "hybrid"], label="Detection Mode", value="hybrid")
            ],
            outputs="text",
            title="Audio Steganography Detector",
            description="Upload a WAV file (16kHz mono) and select detection mode: CNN, Autoencoder, or Hybrid.",
        )
        gui.launch(server_port=7860)

if __name__ == "__main__":
    main()
''' 

import argparse
import gradio as gr
from classifier import AudioClassifier
import os

# Initialize classifier
classifier = AudioClassifier()

# Shared prediction function    
def detect_stego(audio_path, mode):
    predicted_class, reconstruction_error, denoised_path = classifier.predict(audio_path, mode=mode)
    class_label = "Stego" if predicted_class == 1 else "Cover"
    result_text = f"Mode: {mode.upper()}\nPrediction: {class_label}\nReconstruction Error: {reconstruction_error:.6f}"
    
    # Return denoised audio only if stego detected
    if denoised_path and os.path.exists(denoised_path):
        return result_text, denoised_path
    else:
        return result_text, None

def main():

    parser = argparse.ArgumentParser(description="Audio Steganography Detection")
    parser.add_argument("--file", type=str, help="Path to input WAV file")
    parser.add_argument("--mode", type=str, choices=["cnn", "autoencoder", "hybrid"], default="hybrid", help="Detection mode")
    parser.add_argument("--nogui", action="store_true", help="Run without launching the Gradio web app")

    args = parser.parse_args()

    if args.nogui:
        if not args.file or not os.path.isfile(args.file):
            print("❌ Please provide a valid file path using --file when --nogui is set.")
            return
        result_text, denoised_path = detect_stego(args.file, args.mode)
        print("✅ Classification result:\n", result_text)
        if denoised_path:
            print(f"🎧 Denoised audio saved at: {denoised_path}")

        # Optional: write to output.txt
        with open("output.txt", "w") as f:
            f.write(result_text + "\n")
            if denoised_path:
                f.write(f"Denoised file: {denoised_path}\n")

    else:
        # Run Gradio GUI
        gui = gr.Interface(
            fn=detect_stego,
            inputs=[
                gr.Audio(type="filepath", label="Upload WAV Audio"),
                gr.Radio(["cnn", "autoencoder", "hybrid"], label="Detection Mode", value="hybrid")
            ],
            outputs=[
                gr.Text(label="Prediction"),
                gr.Audio(label="Denoised Audio (if stego)")
            ],
            title="Audio Steganography Detector",
            description="Upload a WAV file (16kHz mono) and select detection mode. If Stego is detected, the denoised audio is returned."
        )
        gui.launch(server_port=7860)

if __name__ == "__main__":
    main()
