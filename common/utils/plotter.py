"""
Variance Adaptor Visualization Module
Generates and saves the required:
- Predicted Duration plot
- Predicted Pitch (F0) plot
- Predicted Energy plot
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_duration(phonemes, durations, save_path, title="Duration Prediction"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 4))
    x = np.arange(len(phonemes))
    plt.bar(x, durations, color="royalblue", edgecolor="black", alpha=0.8)
    plt.xticks(x, phonemes, rotation=45, ha="right", fontsize=9)
    plt.xlabel("Phonemes / Tokens")
    plt.ylabel("Duration (Frames)")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pitch(pitch, save_path, title="Pitch (F0) Prediction"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(pitch, color="crimson", linewidth=2, label="Predicted F0")
    plt.xlabel("Frame Index")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_energy(energy, save_path, title="Energy Prediction"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(energy, color="forestgreen", linewidth=2, label="Predicted Energy")
    plt.xlabel("Frame Index")
    plt.ylabel("Magnitude / Norm")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

