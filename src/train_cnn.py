"""
Training and Evaluation Pipeline for Deep Learning Models:
- Trains SIM-CNN (from scratch) and TL-CNN (Transfer Learning)
- Evaluates Convergence, BCE Loss, Accuracy, Precision, Recall
- Plots Training/Validation Curves (mirrors CS229 Figure 6)
- Evaluates Threshold Scanning and Confusion Matrices
"""

import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import load_dataset_subset

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    from src.models_cnn import SimpleCNN, TransferLearningCNN, count_trainable_parameters
except ImportError:
    torch = None


if torch is not None:
    class AerialShipDataset(Dataset):
        """PyTorch Dataset wrapper for Aerial Images with optional augmentations."""
        def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
            self.images = images
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            img = self.images[idx]
            label = self.labels[idx]

            # Convert numpy array (H, W, C) [0..255] to Tensor (C, H, W) [0..1]
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            if self.transform:
                img_tensor = self.transform(img_tensor)

            return img_tensor, torch.tensor(label, dtype=torch.float32)


def get_data_transforms():
    """Defines training and validation data augmentation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    device: str = "cpu"
) -> dict:
    """
    Trains a CNN model and logs training/validation metrics across epochs.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    print(f"\n--- Training {model_name} on {device} ({epochs} epochs) ---")
    trainable_p, total_p = count_trainable_parameters(model)
    print(f"Parameters: {trainable_p:,} trainable / {total_p:,} total")

    best_val_loss = float('inf')
    best_weights_path = f"models/{model_name.lower().replace('-', '_')}.pth"

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(-1)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_train += (preds == lbls).sum().item()
            total_train += lbls.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = (correct_train / max(1, total_train)) * 100

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs).squeeze(-1)
                loss = criterion(outputs, lbls)
                val_running_loss += loss.item() * imgs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == lbls).sum().item()
                total_val += lbls.size(0)

        epoch_val_loss = val_running_loss / max(1, total_val)
        epoch_val_acc = (correct_val / max(1, total_val)) * 100
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:5.1f}% | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:5.1f}%")

    print(f"[OK] Model '{model_name}' saved to '{best_weights_path}' (Best Val Loss: {best_val_loss:.4f})")
    return history


def evaluate_cnn(model, loader, device: str = "cpu", threshold: float = 0.5) -> dict:
    """Evaluate a trained CNN on a held-out set using classification metrics."""
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device)).squeeze(-1)
            y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def plot_cnn_training_comparison(sim_hist: dict, tl_hist: dict, save_path: str = "models/cnn_training_curves.png"):
    """
    Plots training loss and accuracy comparison for 30/15 epochs mirroring Figure 6 from CS229 paper.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(sim_hist["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    ax1.plot(epochs, sim_hist["train_loss"], 'r-', label="SIM-CNN Loss (Train)", lw=2)
    ax1.plot(epochs, tl_hist["train_loss"], 'b-', label="TL-CNN Loss (Train)", lw=2)
    ax1.set_title("Binary Cross-Entropy Loss vs Epochs (CS229 Fig 6)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Binary Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy comparison
    ax2.plot(epochs, sim_hist["train_acc"], 'r--', label="SIM-CNN Accu (Train)", lw=2)
    ax2.plot(epochs, tl_hist["train_acc"], 'b--', label="TL-CNN Accu (Train)", lw=2)
    ax2.set_title("Training Accuracy vs Epochs (CS229 Fig 6)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"CNN comparison curves saved to '{save_path}'")


def run_cnn_training_pipeline(data_dir: str = "data/sample_dataset", epochs: int = 15, batch_size: int = 16):
    """Loads dataset subset and trains both SIM-CNN and TL-CNN."""
    if torch is None:
        print("PyTorch is not available. Please install PyTorch first.")
        return

    if os.path.isdir(os.path.join(data_dir, "train")):
        # A prepared subset keeps the paper-style split completely isolated.
        train_imgs, train_lbls, _ = load_dataset_subset(os.path.join(data_dir, "train"), max_samples_per_class=None)
        val_imgs, val_lbls, _ = load_dataset_subset(os.path.join(data_dir, "val"), max_samples_per_class=None)
        test_imgs, test_lbls, _ = load_dataset_subset(os.path.join(data_dir, "test"), max_samples_per_class=None)
    else:
        images, labels, _ = load_dataset_subset(data_dir=data_dir, max_samples_per_class=None)
        # Fallback for a flat folder: leakage-safe, reproducible 60/20/20 split.
        train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
            images, labels, test_size=0.4, stratify=labels, random_state=42
        )
        val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
            temp_imgs, temp_lbls, test_size=0.5, stratify=temp_lbls, random_state=42
        )

    train_tf, val_tf = get_data_transforms()

    train_dataset = AerialShipDataset(train_imgs, train_lbls, transform=train_tf)
    val_dataset = AerialShipDataset(val_imgs, val_lbls, transform=val_tf)
    test_dataset = AerialShipDataset(test_imgs, test_lbls, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Train SIM-CNN
    sim_model = SimpleCNN()
    sim_history = train_single_model("SIM-CNN", sim_model, train_loader, val_loader, epochs=epochs, lr=0.0005, device=device)

    # 2. Train TL-CNN (DenseNet-169 transfer-learning backbone)
    tl_model = TransferLearningCNN(backbone="densenet169", pretrained=True)
    tl_history = train_single_model("TL-CNN", tl_model, train_loader, val_loader, epochs=epochs, lr=0.001, device=device)

    # Evaluate the best checkpoint on the untouched test split.
    sim_model.load_state_dict(torch.load("models/sim_cnn.pth", map_location=device))
    tl_model.load_state_dict(torch.load("models/tl_cnn.pth", map_location=device))
    sim_metrics = evaluate_cnn(sim_model.to(device), test_loader, device=device)
    tl_metrics = evaluate_cnn(tl_model.to(device), test_loader, device=device)

    # 3. Plot Comparison Curves
    plot_cnn_training_comparison(sim_history, tl_history)

    # Save summary
    cnn_results = {
        "SIM-CNN": {
            "final_train_acc": sim_history["train_acc"][-1],
            "final_val_acc": sim_history["val_acc"][-1],
            "final_train_loss": sim_history["train_loss"][-1],
            "final_val_loss": sim_history["val_loss"][-1],
            "test_metrics": sim_metrics
        },
        "TL-CNN": {
            "final_train_acc": tl_history["train_acc"][-1],
            "final_val_acc": tl_history["val_acc"][-1],
            "final_train_loss": tl_history["train_loss"][-1],
            "final_val_loss": tl_history["val_loss"][-1],
            "test_metrics": tl_metrics
        }
    }

    with open("models/cnn_results.json", "w") as f:
        json.dump(cnn_results, f, indent=4)

    print("\n[OK] Deep Learning pipeline completed successfully.")
    return cnn_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate SIM-CNN and DenseNet-169.")
    parser.add_argument("--data-dir", default="data/subset_6000")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run_cnn_training_pipeline(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
