'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from models.wavecnn import WaveCNN
from data.loader import AudioDataset
from collections import Counter
import numpy as np

def train_wavecnn(audio_dir, labels_file, epochs=10, batch_size=16, learning_rate=0.0005, val_split=0.2):
    # Load dataset
    dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=64000)  # 1 second of 16kHz audio
    
    # Count class distribution
    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    print("Class distribution:", class_counts)

    # Calculate class weights (inversely proportional to frequency)
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / class_counts[0], total / class_counts[1]], dtype=torch.float32)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model and training setup
    model = WaveCNN().to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, val_accuracies = [], [], []

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Normalize
            inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / (inputs.std(dim=1, keepdim=True) + 1e-5)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / (inputs.std(dim=1, keepdim=True) + 1e-5)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_true.extend(labels.cpu().tolist())

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Final classification report
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=["Clean", "Stego"]))

    # Save model
    torch.save(model.state_dict(), "models/wavecnn_trained.pth")
    print("Model saved to models/wavecnn_trained.pth")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_wavecnn("audio", "data/labels.csv")

''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from models.wavecnn import WaveCNN
from data.loader import AudioDataset
from collections import Counter
import numpy as np

def train_wavecnn(audio_dir, labels_file, epochs=10, batch_size=16, learning_rate=0.0005, val_split=0.2):
    # Load dataset using 1-second chunks (16000 samples)
    dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=16000)
    
    # Count class distribution
    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    print("Class distribution:", class_counts)

    # Calculate class weights
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / class_counts[0], total / class_counts[1]], dtype=torch.float32)

    # Split into train/val sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, loss, optimizer
    model = WaveCNN().to(device)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.unsqueeze(1)  # [B, 1, 16000]
          

            inputs = (inputs - inputs.mean(dim=2, keepdim=True)) / (inputs.std(dim=2, keepdim=True) + 1e-5)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)  # [B, 1, 16000]
                inputs = (inputs - inputs.mean(dim=2, keepdim=True)) / (inputs.std(dim=2, keepdim=True) + 1e-5)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_true.extend(labels.cpu().tolist())

        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Final classification report
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=["Clean", "Stego"]))

    torch.save(model.state_dict(), "models/wavecnn_trained.pth")
    print("✅ WaveCNN model saved to models/wavecnn_trained.pth")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.title('Validation Accuracy over Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_wavecnn("audio", "data/labels.csv")
