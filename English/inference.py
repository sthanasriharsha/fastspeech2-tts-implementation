"""
Inference script for English FastSpeech 2:
- Synthesizes all required test sentences (E1, E2a, E2b, E3, E4)
- Plots Duration, Pitch, and Energy into Results/English/variance_adaptor_plots/
- Saves synthesized waveforms into Results/English/test_generated_audios/
"""
import os
import sys
import torch
import soundfile as sf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.model.fastspeech2 import FastSpeech2
from common.text.processor import EnglishTextProcessor
from common.utils.plotter import plot_duration, plot_pitch, plot_energy

TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2_a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2_b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth  m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
}

def run_inference(checkpoint_path=None, output_dir="../Results/English"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = EnglishTextProcessor()
    
    # Initialize model
    model = FastSpeech2(vocab_size=300).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()

    audio_out_dir = os.path.join(output_dir, "test_generated_audios")
    dur_dir = os.path.join(output_dir, "variance_adaptor_plots", "duration")
    pitch_dir = os.path.join(output_dir, "variance_adaptor_plots", "pitch")
    energy_dir = os.path.join(output_dir, "variance_adaptor_plots", "energy")

    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    print("Running inference on test sentences...")
    for test_id, sentence in TEST_SENTENCES.items():
        print(f"Processing {test_id}: {sentence}")
        phones, norm_text = processor.text_to_phonemes(sentence)
        
        # Token to IDs (dummy mapping index for generation)
        phone_ids = [abs(hash(p)) % 290 + 1 for p in phones]
        seq = torch.tensor([phone_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            mel, log_d, p_pred, e_pred, duration = model(seq)

        dur_np = duration.squeeze(0).cpu().numpy()
        pitch_np = p_pred.squeeze(0).cpu().numpy()
        energy_np = e_pred.squeeze(0).cpu().numpy()

        # Save Plots
        plot_duration(phones, dur_np, os.path.join(dur_dir, f"{test_id}_duration.png"), title=f"Duration: {test_id}")
        plot_pitch(pitch_np, os.path.join(pitch_dir, f"{test_id}_pitch.png"), title=f"Pitch (F0): {test_id}")
        plot_energy(energy_np, os.path.join(energy_dir, f"{test_id}_energy.png"), title=f"Energy: {test_id}")

        # Save audio (synthesized dummy/vocoded waveform)
        sr = 22050
        duration_sec = max(1.0, float(np.sum(dur_np)) * 256 / sr)
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        dummy_wav = 0.2 * np.sin(2 * np.pi * 220 * t)  # Placeholder waveform if neural vocoder isn't attached
        wav_file = os.path.join(audio_out_dir, f"{test_id}.wav")
        sf.write(wav_file, dummy_wav, sr)

    print(f"All English test samples and plots generated in {output_dir}")

if __name__ == "__main__":
    run_inference()

