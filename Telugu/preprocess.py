"""
Preprocessing entry point for Telugu (IndicTTS).
"""
import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.utils.subsample import subsample_dataset

def main(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["path"]["raw_path"]
    preprocessed_path = config["path"]["preprocessed_path"]
    os.makedirs(preprocessed_path, exist_ok=True)

    print(f"Preprocessing Telugu ({config['dataset']}) dataset...")
    if os.path.exists(raw_path):
        train_files, val_files = subsample_dataset(raw_path, target_train_sec=18000, target_val_sec=3600)
    else:
        print(f"Directory {raw_path} does not exist yet. Please download IndicTTS Telugu dataset first.")

if __name__ == "__main__":
    cfg = sys.argv[2] if len(sys.argv) > 2 else "Telugu/config.yaml"
    main(cfg)

