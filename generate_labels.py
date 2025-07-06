# import os
# import csv

# cover_dir = "audio/cover"
# stego_dir = "audio/stego"
# output_file = "data/labels.csv"

# with open(output_file, 'w', newline='') as f:
#     writer = csv.writer(f)
    
#     for file in os.listdir(cover_dir):
#         if file.endswith(".wav"):
#             writer.writerow([f"cover/{file}", 0])
    
#     for file in os.listdir(stego_dir):
#         if file.endswith(".wav"):
#             writer.writerow([f"stego/{file}", 1])

# print(f"Labels file written to {output_file}")

import os
import csv
import random

cover_dir = "audio/cover"
stego_dir = "audio/stego"
output_file = "data/labels.csv"

def get_stego_label(filename):
    if '25' in filename:
        return 1
    elif '50' in filename:
        return 2
    elif '75' in filename:
        return 3
    elif '100' in filename:
        return 4
    else:
        return -1  # Unknown label

data_rows = []

# Add cover files with label 0
for file in os.listdir(cover_dir):
    if file.endswith(".wav"):
        data_rows.append([f"cover/{file}", 0])

# Add stego files with corresponding label
for file in os.listdir(stego_dir):
    if file.endswith(".wav"):
        label = get_stego_label(file)
        if label != -1:
            data_rows.append([f"stego/{file}", label])
        else:
            print(f"⚠️ Skipping file (no valid label found): {file}")

# Shuffle the rows
random.shuffle(data_rows)

# Write to CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data_rows)

print(f" Labels file with {len(data_rows)} entries written to: {output_file}")

