"""
Script to filter and subsample audio datasets to:
- Exactly ~5 hours of training data
- Exactly ~1 hour of validation data
- Strict zero-overlap check
"""
import os
import glob
import random
import soundfile as sf

def subsample_dataset(audio_dir: str, target_train_sec=18000, target_val_sec=3600, seed=42):
    random.seed(seed)
    wav_files = glob.glob(os.path.join(audio_dir, "**", "*.wav"), recursive=True)
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return [], []

    print(f"Found {len(wav_files)} total audio files. Inspecting durations...")
    valid_entries = []
    
    for wav_path in wav_files:
        try:
            info = sf.info(wav_path)
            duration = info.duration
            # Keep utterances between 1.5s and 10.0s for stable TTS training
            if 1.5 <= duration <= 10.0:
                valid_entries.append((wav_path, duration))
        except Exception:
            continue

    random.shuffle(valid_entries)

    train_files, val_files = [], []
    train_dur, val_dur = 0.0, 0.0

    for wav_path, dur in valid_entries:
        if train_dur < target_train_sec:
            train_files.append(wav_path)
            train_dur += dur
        elif val_dur < target_val_sec:
            val_files.append(wav_path)
            val_dur += dur
        else:
            break

    print(f"Subsampled Train: {len(train_files)} files ({train_dur/3600:.2f} hours)")
    print(f"Subsampled Val:   {len(val_files)} files ({val_dur/3600:.2f} hours)")
    
    # Assert zero overlap
    overlap = set(train_files).intersection(set(val_files))
    assert len(overlap) == 0, "Error: Overlap detected between train and val sets!"
    print("Zero-overlap verification passed.")

    return train_files, val_files

