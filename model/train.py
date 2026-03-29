"""
Trains a MiniVGG CNN on the MRL Eye Dataset for drowsiness detection.

Features:
- Mixed precision training (AMP) for performance.
- Grayscale conversion and validation tracking.
- Saves the model with the lowest validation loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm

# Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
DATA_DIR = 'D:/University/Junior/HLS/VGG_Accelerator/data/MRL' 
MODEL_SAVE_PATH = 'minivgg.pth'
NUM_WORKERS = 8

class MiniVGG(nn.Module):
    """
    A lightweight VGG-style CNN designed for 32x32 grayscale inputs.
    Output: 2 classes (Open/Closed).
    """
    def __init__(self):
        super(MiniVGG, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.classifier = nn.Linear(64 * 4 * 4, 2) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    # Optimization setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f"Training on: {device} (Optimized)")

    # Data transformation: Grayscale -> Tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found.")
        return

    # Load Datasets
    train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{DATA_DIR}/val", transform=transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Class Mapping: {train_dataset.class_to_idx}") 

    # Model and Training setup
    model = MiniVGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda') 
    best_val_loss = float('inf') 

    print("\n--- Starting Training ---")

    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Logging
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            print(f"   >>> Improvement! Saving model...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            best_val_loss = avg_val_loss

    print(f"\nDone. Best model saved to: {os.path.abspath(MODEL_SAVE_PATH)}")

if __name__ == "__main__":
    main()