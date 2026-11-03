# FastSpeech 2 - English & Telugu Text-to-Speech Implementation

This repository contains the complete implementation, training scripts, evaluation suite, and technical report for FastSpeech 2 text-to-speech synthesis trained on **Svarah (English)** and **IndicTTS (Telugu)** datasets.

---

## 1. Project Directory Structure
```
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

---

## 2. Environment Setup
Install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## 3. Preprocessing & Dataset Subsampling
We filter utterances between 1.5s and 10.0s to ensure clean, stable training:
- **Training Set**: ~5 hours of clean audio
- **Validation Set**: ~1 hour of non-overlapping audio

Run preprocessing for English:
```bash
python English/preprocess.py --config English/config.yaml
```

Run preprocessing for Telugu:
```bash
python Telugu/preprocess.py --config Telugu/config.yaml
```

---

## 4. Model Training
Train English FastSpeech 2:
```bash
python English/train.py --config English/config.yaml
```

Train Telugu FastSpeech 2:
```bash
python Telugu/train.py --config Telugu/config.yaml
```

---

## 5. Synthesis & Variance Adaptor Plotting
To generate the speech samples and variance plots for all test sentences:
```bash
python English/inference.py --config English/config.yaml
python Telugu/inference.py --config Telugu/config.yaml
```

Generated outputs are saved in `Results/` according to the required folder layout.

---

## 6. Model Checkpoints Download Link
If checkpoints exceed GitHub file size limits, they are accessible at:
- English Model Checkpoint: [Google Drive / External Link]
- Telugu Model Checkpoint: [Google Drive / External Link]

