import os
import csv

cover_dir = "audio/cover"
stego_dir = "audio/stego"
output_file = "data/labels.csv"

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    
    for file in os.listdir(cover_dir):
        if file.endswith(".wav"):
            writer.writerow([f"cover/{file}", 0])
    
    for file in os.listdir(stego_dir):
        if file.endswith(".wav"):
            writer.writerow([f"stego/{file}", 1])

print(f"Labels file written to {output_file}")

