import torch
import base64
import os
import zipfile
import gzip
import shutil

def extract_message_from_diff(diff_tensor, embedding_percent=100):
    """
    Extract LSBs from audio difference and decode ASCII message or fallback.
    """
    diff_tensor = diff_tensor.squeeze().cpu()
    num_samples = int(len(diff_tensor) * (embedding_percent / 100.0))
    target_samples = diff_tensor[:num_samples]

    # Scale to int16 and get LSB
    int_wave = (target_samples * 32768).short()
    lsb_bits = (int_wave & 1).tolist()
    message_bits = ''.join(str(b) for b in lsb_bits)

    # Convert bits to bytes
    byte_array = bytearray()
    for i in range(0, len(message_bits) - 7, 8):
        byte = message_bits[i:i + 8]
        byte_array.append(int(byte, 2))

    try:
        message = byte_array.decode("ascii")
    except UnicodeDecodeError:
        try:
            message = base64.b64encode(byte_array).decode("utf-8")
            message = "[BASE64 ENCODED BINARY]\n" + message
        except Exception:
            message = "[UNREADABLE BINARY DATA]"

    return message_bits, message


def fix_base64_padding(b64_string):
    """Ensure proper Base64 padding."""
    b64_string = b64_string.strip()
    missing_padding = len(b64_string) % 4
    if missing_padding:
        b64_string += '=' * (4 - missing_padding)
    return b64_string


def extract_archive(file_path, extract_dir):
    """
    Detect and extract common archive formats (zip, gzip).
    Returns True if extraction was successful, else False.
    """
    try:
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[INFO] Extracted ZIP archive to {extract_dir}")
            return True
        elif file_path.endswith('.gz'):
            # For gzip, decompress to same folder
            decompressed_path = os.path.join(extract_dir, os.path.basename(file_path).replace('.gz', ''))
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"[INFO] Extracted GZIP file to {decompressed_path}")
            return True
    except Exception as e:
        print(f"[WARNING] Archive extraction failed: {e}")
    return False


def decode_base64_payload(b64_data, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix padding
    b64_data = fix_base64_padding(b64_data)

    try:
        binary_data = base64.b64decode(b64_data)
    except Exception as e:
        print(f"[ERROR] Failed to decode Base64: {e}")
        return None

    bin_path = os.path.join(output_dir, "decoded_payload.bin")
    with open(bin_path, "wb") as f:
        f.write(binary_data)
    print(f"[INFO] Binary payload saved to: {bin_path}")

    # Try to extract archive if possible
    extracted = extract_archive(bin_path, output_dir)
    if extracted:
        # If extraction successful, return folder path
        return output_dir

    # Try to decode as UTF-8 text and save
    try:
        decoded_text = binary_data.decode("utf-8")
        text_path = os.path.join(output_dir, "decoded_message.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(decoded_text)
        print(f"[INFO] UTF-8 text saved to: {text_path}")
        return text_path
    except UnicodeDecodeError:
        print("[INFO] Binary data is not UTF-8 text.")

    return bin_path
