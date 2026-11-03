# FastSpeech 2: English & Telugu Text-to-Speech Implementation

This repository contains the complete implementation, training pipeline, generated speech samples, variance adaptor visualizations, and technical report for FastSpeech 2 Text-to-Speech synthesis for **English (Svarah)** and **Telugu (IndicTTS Telugu)**.

---

## 1. Directory Structure

The repository strictly follows the required assignment layout:

```
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
│       └── fastspeech2_english.pth
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
│       └── fastspeech2_telugu.pth
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
│   ├── Technical_Report.pdf
│   └── Technical_Report.md
├── fastspeech2_kaggle_pipeline.ipynb
├── README.md
└── requirements.txt
```

---

## 2. Model Checkpoints & Accessible Working Download Links

In accordance with **Submission Requirement 4.b** regarding checkpoint file sizes:

| Model | Architecture | Training Dataset | Checkpoint Path | Accessible Download Link |
|---|---|---|---|---|
| **English Model** | FastSpeech 2 + Variance Adaptor | AI4Bharat Svarah (~5h train / ~1h val) | `English/checkpoints/fastspeech2_english.pth` | [Download English Checkpoint (Google Drive)](https://drive.google.com/drive/folders/1N-ydMQxhFcuggNf132xJ39q3iV1OCSMM?usp=sharing) |
| **Telugu Model** | FastSpeech 2 + Variance Adaptor | IndicTTS Telugu (~5h train / ~1h val) | `Telugu/checkpoints/fastspeech2_telugu.pth` | [Download Telugu Checkpoint (Google Drive)](https://drive.google.com/drive/folders/1N-ydMQxhFcuggNf132xJ39q3iV1OCSMM?usp=sharing) |

> **Note for Evaluators**: If downloading checkpoints from external storage, place `fastspeech2_english.pth` into `English/checkpoints/` and `fastspeech2_telugu.pth` into `Telugu/checkpoints/`.

---

## 3. Dataset Preprocessing & Subsampling Strategy

- **English Dataset**: [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Telugu Dataset**: [SPRINGLab IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
- **Audio Criteria**: Utterance durations filtered strictly between **1.5 seconds and 10.0 seconds** to avoid silence corruption and memory overflow.
- **Data Partitioning**:
  - **Training Split**: Exactly ~5.0 hours of audio (18,000 seconds)
  - **Validation Split**: Exactly ~1.0 hour of audio (3,600 seconds)
  - **Zero Overlap**: Disjoint utterance-ID sets verified programmatically (`len(train_ids ∩ val_ids) == 0`).

---

## 4. Test Sentences & Evaluation Scope

### English Test Sentences:
- **E1**: *The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.* (Tests polysyllabic stress and duration stability).
- **E2**:
  - **(a)** *The orchestra's conduct during the ceremony was praised by the audience.* (Noun stress: `CON-duct`).
  - **(b)** *The researchers will conduct an experiment to evaluate the proposed method.* (Verb stress: `con-DUCT`).
- **E3**: *The rhyth  m of the orchestra changed unexpectedly.* (Tests whitespace anomaly robustness).
- **E4**: *The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.* (Tests time/date text normalization and code-mixed Telugu loanword *andaroo*).

### Telugu Test Sentences:
- **T1**: *న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.* (Tests long compound and medical transliteration).
- **T2**:
  - **(a)** *తాడుపై అతని పట్టు చాలా బలంగా ఉంది.* (Homograph context 1: grip/hold).
  - **(b)** *ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.* (Homograph context 2: silk).
- **T3**: *పులి అడ విలో నిశ్శబ్దంగా నడిచింది.* (Tests word splitting artifacts and syllable pauses).
- **T4**: *ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.* (Tests date, time, and numerical text normalization in Telugu).

---

## 5. Quickstart & Reproducibility

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Preprocessing
```bash
python English/preprocess.py --config English/config.yaml
python Telugu/preprocess.py --config Telugu/config.yaml
```

### 3. Model Training
```bash
python English/train.py --config English/config.yaml
python Telugu/train.py --config Telugu/config.yaml
```

### 4. Run Inference & Generate Variance Adaptor Plots
```bash
python English/inference.py --config English/config.yaml
python Telugu/inference.py --config Telugu/config.yaml
```

Generated outputs will populate `Results/English/` and `Results/Telugu/` with:
- Synthesized `.wav` audio files in `test_generated_audios/`
- Frame-level duration bar plots in `variance_adaptor_plots/duration/`
- Continuous pitch ($F_0$) curves in `variance_adaptor_plots/pitch/`
- Energy magnitude curves in `variance_adaptor_plots/energy/`

---

## 6. Technical Report
A comprehensive report analyzing model architecture, training loss convergence, variance adaptor dynamics, comparative observations, and failure modes is provided in:
📄 [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf)
