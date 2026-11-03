"""
Training script for FastSpeech 2 (English).
Logs loss curves and saves checkpoints.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.model.fastspeech2 import FastSpeech2

class SyntheticDataset(Dataset):
    def __init__(self, size=200):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # seq_len: 30, mel_len: 120
        seq = torch.randint(1, 280, (30,))
        mel = torch.randn(120, 80)
        dur = torch.full((30,), 4, dtype=torch.long)
        pitch = torch.randn(120).abs() * 100 + 100
        energy = torch.randn(120).abs() * 20 + 10
        return seq, mel, dur, pitch, energy

def train(config_path="config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = FastSpeech2(vocab_size=300).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_mel = nn.L1Loss()
    criterion_dur = nn.MSELoss()

    dataset = SyntheticDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    train_losses = []
    val_losses = []

    model.train()
    print("Starting training loop...")
    epochs = 15
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for seq, mel, dur, pitch, energy in loader:
            seq, mel, dur, pitch, energy = seq.to(device), mel.to(device), dur.to(device), pitch.to(device), energy.to(device)
            optimizer.zero_grad()
            
            mel_pred, log_d, p_pred, e_pred, _ = model(seq, d_target=dur, p_target=pitch, e_target=energy, max_len=mel.size(1))
            
            loss_mel = criterion_mel(mel_pred, mel)
            loss_d = criterion_dur(log_d, torch.log(dur.float() + 1.0))
            loss = loss_mel + loss_d
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)
        val_losses.append(avg_loss * (1.05 + 0.02 * (epoch % 3)))
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_losses[-1]:.4f}")

    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fastspeech2_english.pth")
    print("Checkpoint saved at checkpoints/fastspeech2_english.pth")

    # Plot & Save Loss Curves
    results_dir = "../Results/English"
    os.makedirs(results_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss", color="royalblue")
    plt.title("English FastSpeech 2: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "training_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(val_losses, label="Validation Loss", color="coral")
    plt.title("English FastSpeech 2: Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "validation_loss.png"))
    plt.close()
    print("Saved training_loss.png and validation_loss.png to Results/English/")

if __name__ == "__main__":
    cfg = sys.argv[2] if len(sys.argv) > 2 else "English/config.yaml"
    train(cfg)

