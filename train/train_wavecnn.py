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


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# from models.wavecnn import WaveCNN
# from data.loader import AudioDataset
# from collections import Counter
# import numpy as np

# def train_wavecnn(audio_dir, labels_file, epochs=10, batch_size=16, learning_rate=0.0005, val_split=0.2):
#     # Load dataset using 1-second chunks (16000 samples)
#     dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=16000)
    
#     # Count class distribution
#     all_labels = [label for _, label in dataset]
#     class_counts = Counter(all_labels)
#     print("Class distribution:", class_counts)

#     # Calculate class weights
#     total = sum(class_counts.values())
#     class_weights = torch.tensor([total / class_counts[0], total / class_counts[1]], dtype=torch.float32)

#     # Split into train/val sets
#     val_size = int(len(dataset) * val_split)
#     train_size = len(dataset) - val_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     # Device setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Model, loss, optimizer
#     model = WaveCNN().to(device)
#     class_weights = class_weights.to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     train_losses, val_losses, val_accuracies = [], [], []

#     for epoch in range(epochs):
#         model.train()
#         total_loss = 0.0
#         correct = 0
#         total = 0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
            
#             inputs = inputs.unsqueeze(1)  # [B, 1, 16000]
          

#             inputs = (inputs - inputs.mean(dim=2, keepdim=True)) / (inputs.std(dim=2, keepdim=True) + 1e-5)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         train_accuracy = 100 * correct / total
#         train_losses.append(total_loss)

#         # Validation
#         model.eval()
#         val_loss = 0.0
#         val_correct = 0
#         val_total = 0
#         all_preds, all_true = [], []

#         with torch.no_grad():
#             for inputs, labels in val_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 inputs = inputs.unsqueeze(1)  # [B, 1, 16000]
#                 inputs = (inputs - inputs.mean(dim=2, keepdim=True)) / (inputs.std(dim=2, keepdim=True) + 1e-5)

#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()

#                 _, predicted = torch.max(outputs.data, 1)
#                 val_total += labels.size(0)
#                 val_correct += (predicted == labels).sum().item()

#                 all_preds.extend(predicted.cpu().tolist())
#                 all_true.extend(labels.cpu().tolist())

#         val_accuracy = 100 * val_correct / val_total
#         val_losses.append(val_loss)
#         val_accuracies.append(val_accuracy)

#         print(f"Epoch {epoch + 1}/{epochs}, "
#               f"Train Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
#               f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

#     # Final classification report
#     print("\nClassification Report:")
#     print(classification_report(all_true, all_preds, target_names=["Clean", "Stego"]))

#     torch.save(model.state_dict(), "models/wavecnn_trained.pth")
#     print(" WaveCNN model saved to models/wavecnn_trained.pth")
    
#     # Plot
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Val Loss')
#     plt.legend()
#     plt.title('Loss over Epochs')

#     plt.subplot(1, 2, 2)
#     plt.plot(val_accuracies, label='Val Accuracy')
#     plt.legend()
#     plt.title('Validation Accuracy over Epochs')

#     plt.tight_layout()
#     plt.show()
    
# if __name__ == '__main__':
#     train_wavecnn("audio", "data/labels.csv")

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
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_wavecnn(audio_dir, labels_file, epochs=15, batch_size=8, learning_rate=0.001, val_split=0.2):
    # Load dataset with 4-second audio samples (176400 samples)
    dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=176400)
    
    # Count class distribution
    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    print("Class distribution:", class_counts)

    # Calculate class weights for all classes
    total = sum(class_counts.values())
    class_weights = torch.tensor([total / class_counts[c] for c in sorted(class_counts.keys())], dtype=torch.float32)

    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement before stopping
    best_val_acc = 0.0
    epochs_without_improvement = 0

    # Initialize model
    model = WaveCNN(num_classes=5).to(device)  # Make sure your model supports multi-class output
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # [B, 1, 176400]

            # Normalize per sample
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
                inputs = inputs.unsqueeze(1)  # [B, 1, 176400]
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
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                # Check for improvement
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/wavecnn_best.pth")
            print(f"✅ Best model saved (Val Acc: {best_val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ No improvement for {epochs_without_improvement} epoch(s)")
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break
    # Final classification report
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=["Cover", "25%", "50%", "75%", "100%"]))


    # Plot training/validation loss and accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_wavecnn("audio", "data/labels.csv")


''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.wavecnn import WaveCNN
from data.loader import AudioDataset
from collections import Counter
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

def train_wavecnn(audio_dir, labels_file, epochs=30, batch_size=16, learning_rate=0.0005, val_split=0.2, use_focal_loss=True):
    dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=16000)
    
    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    print("Class distribution:", class_counts)

    total = sum(class_counts.values())
    class_weights = torch.tensor([total / class_counts[0], total / class_counts[1]], dtype=torch.float32)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Weighted sampling for class balance
    sample_weights = [1.0 / class_counts[label] for _, label in train_dataset]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveCNN().to(device)
    class_weights = class_weights.to(device)
    
    criterion = FocalLoss(gamma=2.0, weight=class_weights) if use_focal_loss else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    all_preds, all_true = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
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

        train_acc = 100 * correct / total
        train_losses.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        val_preds, val_true = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                inputs = (inputs - inputs.mean(dim=2, keepdim=True)) / (inputs.std(dim=2, keepdim=True) + 1e-5)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_preds.extend(predicted.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_wavecnn.pth")

        all_preds = val_preds
        all_true = val_true

    # Final report
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=["Clean", "Stego"]))

    # Confusion Matrix
    cm = confusion_matrix(all_true, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Stego"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Plot Loss and Accuracy
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

    print("Best WaveCNN model saved to models/best_wavecnn.pth")

if __name__ == '__main__':
    train_wavecnn("audio", "data/labels.csv")
'''
''' 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.wavecnn import WaveCNN
from data.loader import AudioDataset
from collections import Counter
import numpy as np

def train_wavecnn(audio_dir, labels_file, epochs=30, batch_size=16, learning_rate=0.0005, val_split=0.2):
    dataset = AudioDataset(audio_dir=audio_dir, labels_file=labels_file, sample_len=16000)

    all_labels = [label for _, label in dataset]
    class_counts = Counter(all_labels)
    print("Class distribution:", class_counts)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_labels = [label for _, label in train_dataset]
    class_sample_count = Counter(train_labels)
    weights = [1.0 / class_sample_count[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WaveCNN().to(device)

    class_weights = torch.tensor(
        [len(train_labels)/class_sample_count[0], len(train_labels)/class_sample_count[1]],
        dtype=torch.float32
    ).to(device)
    pos_weight = class_weights[1] / class_weights[0]  # for BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / (inputs.std(dim=1, keepdim=True) + 1e-5)
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # ✅ no unsqueeze here

            outputs = model(inputs)  # ✅ already [B]
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_true = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = (inputs - inputs.mean(dim=1, keepdim=True)) / (inputs.std(dim=1, keepdim=True) + 1e-5)
                inputs = inputs.to(device)
                labels = labels.to(device).float()  # ✅ no unsqueeze

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().tolist())
                val_true.extend(labels.cpu().tolist())

        val_acc = 100 * val_correct / val_total
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/wavecnn_trained.pth")

    print("\nFinal Classification Report:")
    print(classification_report(val_true, val_preds, target_names=["Clean", "Stego"]))

    cm = confusion_matrix(val_true, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Clean", "Stego"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    print("Best model saved to models/wavecnn_trained.pth")

if __name__ == '__main__':
    train_wavecnn("audio", "data/labels.csv")
'''