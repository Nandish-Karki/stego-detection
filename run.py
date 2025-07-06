# import argparse
# import gradio as gr
# from classifier import AudioClassifier
# import os

# # Initialize classifier
# classifier = AudioClassifier()

# # Shared prediction function
# def detect_stego(audio_path, mode):
#     predicted_class, reconstruction_error = classifier.predict(audio_path, mode=mode)
#     class_label = "Stego" if predicted_class == 1 else "Cover"
#     return f"Mode: {mode.upper()}\nPrediction: {class_label}\nReconstruction Error: {reconstruction_error:.6f}"

# def main():
#     parser = argparse.ArgumentParser(description="Audio Steganography Detection")
#     parser.add_argument("--file", type=str, help="Path to input WAV file")
#     parser.add_argument("--mode", type=str, choices=["cnn", "autoencoder", "hybrid"], default="hybrid", help="Detection mode")
#     parser.add_argument("--nogui", action="store_true", help="Run without launching the Gradio web app")

#     args = parser.parse_args()

#     if args.nogui:
#         if not args.file or not os.path.isfile(args.file):
#             print(" Please provide a valid file path using --file when --nogui is set.")
#             return
#         result = detect_stego(args.file, args.mode)
#         print(" Classification result:\n", result)

#         # Optional: write to output.txt
#         with open("output.txt", "w") as f:
#             f.write(result + "\n")
#     else:
#         # Run Gradio GUI
#         gui = gr.Interface(
#             fn=detect_stego,
#             inputs=[
#                 gr.Audio(type="filepath", label="Upload WAV Audio"),
#                 gr.Radio(["cnn", "autoencoder", "hybrid"], label="Detection Mode", value="hybrid")
#             ],
#             outputs="text",
#             title="Audio Steganography Detector",
#             description="Upload a WAV file (16kHz mono) and select detection mode: CNN, Autoencoder, or Hybrid.",
#         )
#         gui.launch(server_port=7860)

# if __name__ == "__main__":
#     main()
 

# import argparse
# import gradio as gr
# from classifier import AudioClassifier
# import os

# # Initialize classifier
# classifier = AudioClassifier()

# # Shared prediction function    
# def detect_stego(audio_path, mode):
#     predicted_class, reconstruction_error, denoised_path ,diff_path= classifier.predict(audio_path, mode=mode)
#     class_label = "Stego" if predicted_class == 1 else "Cover"
#     if reconstruction_error is not None:
#         error_str = f"{reconstruction_error:.6f}"
#     else:
#         error_str = "N/A"
#     result_text = f"Mode: {mode.upper()}\nPrediction: {class_label}\nReconstruction Error: {error_str}"
    
#     # Return denoised audio only if stego detected
#     if denoised_path and diff_path and os.path.exists(denoised_path) and os.path.exists(diff_path):
#         return result_text, denoised_path , diff_path #,message_path,decoded_payload_path
#     else:
#         return result_text, None

# def main():

#     parser = argparse.ArgumentParser(description="Audio Steganography Detection")
#     parser.add_argument("--file", type=str, help="Path to input WAV file")
#     parser.add_argument("--mode", type=str, choices=["cnn", "autoencoder", "hybrid"], default="hybrid", help="Detection mode")
#     parser.add_argument("--nogui", action="store_true", help="Run without launching the Gradio web app")

#     args = parser.parse_args()

#     if args.nogui:
#         if not args.file or not os.path.isfile(args.file):
#             print("Please provide a valid file path using --file when --nogui is set.")
#             return
#         result_text, denoised_path = detect_stego(args.file, args.mode)
#         print(" Classification result:\n", result_text)
#         if denoised_path:
#             print(f"Denoised audio saved at: {denoised_path}")

#         # Optional: write to output.txt
#         with open("output.txt", "w") as f:
#             f.write(result_text + "\n")
#             if denoised_path:
#                 f.write(f"Denoised file: {denoised_path}\n")

#     else:
#         # Run Gradio GUI
#         gui = gr.Interface(
#             fn=detect_stego,
#             inputs=[
#                 gr.Audio(type="filepath", label="Upload WAV Audio"),
#                 gr.Radio(["cnn", "autoencoder", "hybrid"], label="Detection Mode", value="hybrid")
#             ],
#             outputs=[
#                 gr.Text(label="Prediction"),
#                 gr.Audio(label="Denoised Audio (if stego)")
#             ],
#             title="Audio Steganography Detector",
#             description="Upload a WAV file (16kHz mono) and select detection mode. If Stego is detected, the denoised audio is returned."
#         )
#         gui.launch()

# if __name__ == "__main__":
#     main()

import argparse
import gradio as gr
from classifier import AudioClassifier
import os

# Initialize classifier
classifier = AudioClassifier()

# Updated prediction function for multi-class CNN
def detect_stego(audio_path, mode):
    predicted_class,reconstruction_error, denoised_path, diff_path = classifier.predict(audio_path, mode=mode)

    # Class label map for multi-class (edit these labels as needed)
    class_labels = {
        0: "Cover",
        1: "Stego-25%",
        2: "Stego-50%",
        3: "Stego-75%",
        4: "Stego-100%"
    }
    class_label = class_labels.get(predicted_class, f"Class {predicted_class}")
        # Format error
    error_str = f"{reconstruction_error:.6f}" if reconstruction_error is not None else "N/A"

    result_text = f"Mode: {mode.upper()}\nPrediction: {class_label},\nReconstruction Error: {error_str}"

    # Return denoised audio only if path exists
    if denoised_path and diff_path and os.path.exists(denoised_path) and os.path.exists(diff_path):
        return result_text, denoised_path, diff_path
    else:
        return result_text, None, None

def main():
    parser = argparse.ArgumentParser(description="Audio Steganography Detection")
    parser.add_argument("--file", type=str, help="Path to input WAV file")
    parser.add_argument("--mode", type=str, choices=["cnn", "autoencoder", "hybrid"], default="hybrid", help="Detection mode")
    parser.add_argument("--nogui", action="store_true", help="Run without launching the Gradio web app")

    args = parser.parse_args()

    if args.nogui:
        if not args.file or not os.path.isfile(args.file):
            print("Please provide a valid file path using --file when --nogui is set.")
            return
        result_text, denoised_path, _ = detect_stego(args.file, args.mode)
        print("Classification result:\n", result_text)
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
                gr.Audio(label="Denoised Audio (if stego)"),
                gr.Audio(label="Diff Audio (optional)")
            ],
            title="Audio Steganography Detector",
            description="Upload a WAV file (16kHz mono) and select detection mode. Displays multi-class prediction results."
        )
        gui.launch()

if __name__ == "__main__":
    main()
