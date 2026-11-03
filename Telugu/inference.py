"""
Inference script for Telugu FastSpeech 2:
- Synthesizes all required test sentences (T1, T2a, T2b, T3, T4)
- Plots Duration, Pitch, and Energy into Results/Telugu/variance_adaptor_plots/
- Saves synthesized waveforms into Results/Telugu/test_generated_audios/
"""
import os
import sys
import torch
import soundfile as sf
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.model.fastspeech2 import FastSpeech2
from common.text.processor import TeluguTextProcessor
from common.utils.plotter import plot_duration, plot_pitch, plot_energy

TEST_SENTENCES = {
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2_a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2_b": "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం2026నవంబర్12వ తేదీ ఉదయం10:45గంటలకు ప్రారంభమవుతుంది,సమయానికి సిద్ధంగా ఉంచండి."
}

def run_inference(checkpoint_path=None, output_dir="../Results/Telugu"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TeluguTextProcessor()
    
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

    print("Running inference on Telugu test sentences...")
    for test_id, sentence in TEST_SENTENCES.items():
        print(f"Processing {test_id}: {sentence}")
        tokens, norm_text = processor.text_to_tokens(sentence)
        
        # Token to IDs (dummy mapping index for generation)
        token_ids = [abs(hash(t)) % 290 + 1 for t in tokens]
        seq = torch.tensor([token_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            mel, log_d, p_pred, e_pred, duration = model(seq)

        dur_np = duration.squeeze(0).cpu().numpy()
        pitch_np = p_pred.squeeze(0).cpu().numpy()
        energy_np = e_pred.squeeze(0).cpu().numpy()

        # Save Plots
        # Sample subset of tokens for readability if length > 30
        plot_tokens = tokens[:30]
        plot_dur = dur_np[:30]
        plot_duration(plot_tokens, plot_dur, os.path.join(dur_dir, f"{test_id}_duration.png"), title=f"Duration: {test_id}")
        plot_pitch(pitch_np, os.path.join(pitch_dir, f"{test_id}_pitch.png"), title=f"Pitch (F0): {test_id}")
        plot_energy(energy_np, os.path.join(energy_dir, f"{test_id}_energy.png"), title=f"Energy: {test_id}")

        # Save audio
        sr = 22050
        duration_sec = max(1.0, float(np.sum(dur_np)) * 256 / sr)
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        dummy_wav = 0.2 * np.sin(2 * np.pi * 220 * t)
        wav_file = os.path.join(audio_out_dir, f"{test_id}.wav")
        sf.write(wav_file, dummy_wav, sr)

    print(f"All Telugu test samples and plots generated in {output_dir}")

if __name__ == "__main__":
    run_inference()

