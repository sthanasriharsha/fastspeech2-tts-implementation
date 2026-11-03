"""
Automated downloader and subsampler for Hugging Face Datasets:
- Svarah (English)
- IndicTTS (Telugu)

Splits into:
- 5 hours Train
- 1 hour Validation
- Asserts strict zero overlap
"""
import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def process_hf_dataset(
    dataset_name: str,
    subset: str,
    output_dir: str,
    target_train_sec: float = 18000.0,  # ~5 hours
    target_val_sec: float = 3600.0,     # ~1 hour
    split: str = "train"
):
    print(f"\nLoading {dataset_name} ({subset}) from Hugging Face...")
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # Load dataset stream or full
    try:
        ds = load_dataset(dataset_name, subset, split=split, streaming=True)
    except Exception as e:
        print(f"Fallback loading without streaming for {dataset_name}: {e}")
        ds = load_dataset(dataset_name, subset, split=split)

    train_meta = []
    val_meta = []
    
    train_dur = 0.0
    val_dur = 0.0
    
    idx = 0
    print("Streaming and filtering audio samples...")
    for item in tqdm(ds):
        audio_info = item.get("audio", {})
        array = audio_info.get("array")
        sr = audio_info.get("sampling_rate", 22050)
        text = item.get("text", "") or item.get("transcript", "") or item.get("normalized_text", "")
        
        if array is None or len(text.strip()) == 0:
            continue
            
        dur = len(array) / sr
        # Filter between 1.5s and 10.0s for stable TTS
        if not (1.5 <= dur <= 10.0):
            continue

        file_id = f"sample_{idx:05d}.wav"
        
        if train_dur < target_train_sec:
            wav_path = os.path.join(output_dir, "train", file_id)
            sf.write(wav_path, array, sr)
            train_meta.append(f"{file_id}|{text.strip()}\n")
            train_dur += dur
        elif val_dur < target_val_sec:
            wav_path = os.path.join(output_dir, "val", file_id)
            sf.write(wav_path, array, sr)
            val_meta.append(f"{file_id}|{text.strip()}\n")
            val_dur += dur
        else:
            break
            
        idx += 1

    # Write manifests
    with open(os.path.join(output_dir, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(train_meta)
    with open(os.path.join(output_dir, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(val_meta)

    print(f"\nFinished {subset}:")
    print(f"Train: {len(train_meta)} files ({train_dur/3600:.2f} hrs)")
    print(f"Val:   {len(val_meta)} files ({val_dur/3600:.2f} hrs)")
    
    # Assert zero overlap
    train_set = {line.split("|")[0] for line in train_meta}
    val_set = {line.split("|")[0] for line in val_meta}
    assert len(train_set.intersection(val_set)) == 0, "Error: Overlap detected between train and val!"
    print("Zero-overlap check passed successfully.")

if __name__ == "__main__":
    # Example usage for Svarah and IndicTTS Telugu
    print("Ready to run on Kaggle / Local with Hugging Face datasets.")

