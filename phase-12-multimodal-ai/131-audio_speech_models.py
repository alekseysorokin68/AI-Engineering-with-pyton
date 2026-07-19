"""
131 — Audio & Speech Models: Whisper architecture, audio tokens, speech processing

Темы:
  1. Audio Tokenization (mel spectrograms, audio tokens, codec models)
  2. Whisper Architecture (encoder-decoder, multilingual, multitask)
  3. Speech-to-Text Pipeline (audio preprocessing, CTC/attention decoding)
  4. Text-to-Speech (autoregressive TTS, voice cloning concepts)

Самодостаточный файл — не требует numpy, torch, transformers.
"""

import math
import random
import re
import json
import hashlib
import collections

random.seed(42)


# ---------------------------------------------------------------------------
# 1. Audio Tokenization — mel spectrograms, audio tokens, codecs
# ---------------------------------------------------------------------------
def demo_audio_tokenization():
    """Convert raw audio waveforms into tokens models can process."""
    print("=" * 70)
    print("DEMO 1: Audio Tokenization — mel spectrograms, audio tokens, codecs")
    print("=" * 70)

    sample_rate = 16000
    duration = 0.5
    n_samples = int(sample_rate * duration)

    # --- 1a. Generate a synthetic sine wave ---
    freq = 440.0  # A4 note
    waveform = [math.sin(2 * math.pi * freq * t / sample_rate) for t in range(n_samples)]

    print(f"\n--- 1a. Synthetic waveform ---")
    print(f"  Sample rate: {sample_rate} Hz, Duration: {duration}s, Samples: {n_samples}")
    print(f"  Frequency: {freq} Hz (A4 note)")
    print(f"  First 8 samples: {[f'{v:.4f}' for v in waveform[:8]]}")
    print(f"  Peak amplitude: {max(abs(v) for v in waveform):.4f}")

    # --- 1b. Compute DFT (simplified) for spectrogram ---
    def dft(frame, n_freqs=8):
        N = len(frame)
        magnitudes = []
        for k in range(n_freqs):
            real = sum(frame[n] * math.cos(2 * math.pi * k * n / N) for n in range(N))
            imag = sum(frame[n] * math.sin(2 * math.pi * k * n / N) for n in range(N))
            magnitudes.append(math.sqrt(real**2 + imag**2) / N)
        return magnitudes

    frame_size = 256
    n_mels = 8
    spectrogram = []
    for start in range(0, min(n_samples, frame_size * 4), frame_size):
        frame = waveform[start:start + frame_size]
        mags = dft(frame, n_mels)
        spectrogram.append(mags)

    print(f"\n--- 1b. DFT magnitude spectrum ({len(spectrogram)} frames, {n_mels} freq bins) ---")
    print(f"  Frame size: {frame_size} samples ({frame_size/sample_rate*1000:.0f} ms)")
    for i, mags in enumerate(spectrogram[:4]):
        print(f"  Frame {i}: {[f'{v:.4f}' for v in mags]}")
    peak_bin = n_mels - 1
    peak_freq = peak_bin * sample_rate / frame_size
    print(f"  Peak at bin {peak_bin} ~ {peak_freq:.0f} Hz (matches input {freq} Hz)")

    # --- 1c. Mel scale ---
    def hz_to_mel(hz):
        return 2595 * math.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    print(f"\n--- 1c. Mel scale — maps Hz to perceived pitch ---")
    print(f"  {'Hz':>8}  {'Mel':>10}  {'Note':<20}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*20}")
    notes = [(20, "Low rumble"), (100, "Bass"), (440, "A4 (concert)"),
             (1000, "Soprano C6"), (4000, "High sibilance"), (8000, "Air/breath")]
    for hz, label in notes:
        mel = hz_to_mel(hz)
        print(f"  {hz:8d}  {mel:10.1f}  {label}")

    # --- 1d. Audio tokenizer comparison ---
    print(f"\n--- 1d. Audio tokenization approaches ---")
    print(f"  {'Method':<25} {'Frame':<10} {'Vocab':<12} {'Tokens/s':<12} {'Use case'}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*12} {'-'*25}")
    methods = [
        ("Mel spectrogram", "25ms", "continuous", "~40", "Whisper input"),
        ("EnCodec (Facebook)", "7.5ms", "1024", "~133", "Audio LM tokens"),
        ("SoundStream (Google)", "10ms", "4096", "~100", "Neural codec"),
        ("HuBERT (quantized)", "20ms", "504", "~50", "Speech tokens"),
    ]
    for method, frame, vocab, tps, use in methods:
        print(f"  {method:<25} {frame:<10} {vocab:<12} {tps:<12} {use}")


# ---------------------------------------------------------------------------
# 2. Whisper Architecture — encoder-decoder, multilingual, multitask
# ---------------------------------------------------------------------------
def demo_whisper_architecture():
    """Whisper: an encoder-decoder transformer for speech recognition."""
    print("\n" + "=" * 70)
    print("DEMO 2: Whisper Architecture — encoder-decoder, multilingual")
    print("=" * 70)

    # --- 2a. Whisper model dimensions ---
    models = {
        "tiny":   {"d": 384,  "enc": 4,  "dec": 4,  "heads": 6,  "params": "39M"},
        "base":   {"d": 512,  "enc": 6,  "dec": 6,  "heads": 8,  "params": "74M"},
        "small":  {"d": 768,  "enc": 12, "dec": 12, "heads": 12, "params": "244M"},
        "medium": {"d": 1024, "enc": 24, "dec": 24, "heads": 16, "params": "769M"},
        "large":  {"d": 1280, "enc": 32, "dec": 32, "heads": 20, "params": "1550M"},
    }

    print(f"\n--- 2a. Whisper model variants ---")
    print(f"  {'Model':<10} {'Dim':>6} {'Enc':>5} {'Dec':>5} {'Heads':>7} {'Params':<10}")
    print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*5} {'-'*7} {'-'*10}")
    for name, spec in models.items():
        print(f"  {name:<10} {spec['d']:>6} {spec['enc']:>5} "
              f"{spec['dec']:>5} {spec['heads']:>7} {spec['params']:<10}")

    # --- 2b. Encoder processing ---
    n_mels = 80
    n_frames = 3000
    spec_width = n_frames // 2

    print(f"\n--- 2b. Encoder processing ---")
    print(f"  Input: mel spectrogram {n_mels} x {n_frames}")
    print(f"  After conv layers (stride 2): {n_mels} x {spec_width}")
    print(f"  Output: encoder latent {n_mels} x {spec_width}")
    print(f"  Encoder: {models['base']['enc']} transformer layers, "
          f"d_model={models['base']['d']}")

    # --- 2c. Special tokens ---
    special_tokens = {
        "<|startoftranscript|>": "begin decoding",
        "<|en|>": "English", "<|zh|>": "Chinese",
        "<|ja|>": "Japanese", "<|ru|>": "Russian",
        "<|transcribe|>": "transcription task",
        "<|translate|>": "translation task",
        "<|notimestamps|>": "disable timestamps",
        "<|0.00|>": "timestamp 0.00s",
    }

    print(f"\n--- 2c. Special tokens for multitask + multilingual ---")
    for token, meaning in special_tokens.items():
        print(f"  {token:<30} -> {meaning}")

    # --- 2d. Decoding loop simulation ---
    print(f"\n--- 2d. Autoregressive decoding simulation ---")
    vocab = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>",
             "Hello", "world", ",", " how", " are", " you", "?", "  "]
    sequence = []
    for step in range(8):
        probs = [0.02] * len(vocab)
        if step == 0:
            probs[0] = 0.95
        elif step == 1:
            probs[1] = 0.90
        elif step == 2:
            probs[2] = 0.92
        else:
            probs[3 + (step - 3) % 7] = 0.80
        total = sum(probs)
        probs = [p / total for p in probs]
        idx = probs.index(max(probs))
        sequence.append(vocab[idx])
        print(f"  Step {step}: selected '{vocab[idx]}' (p={max(probs):.3f})")
    print(f"  Final: '{''.join(sequence)}'")


# ---------------------------------------------------------------------------
# 3. Speech-to-Text Pipeline — preprocessing, CTC, attention decoding
# ---------------------------------------------------------------------------
def demo_speech_to_text():
    """Full speech-to-text pipeline from audio to transcript."""
    print("\n" + "=" * 70)
    print("DEMO 3: Speech-to-Text Pipeline — CTC vs attention decoding")
    print("=" * 70)

    # --- 3a. Audio preprocessing steps ---
    print(f"\n--- 3a. Audio preprocessing pipeline ---")
    steps = [
        ("1. Resample", "16kHz mono (if needed)"),
        ("2. Normalize", "peak amplitude = 1.0"),
        ("3. Pre-emphasis", "y'[n] = y[n] - 0.97 * y[n-1]"),
        ("4. Framing", "25ms windows, 10ms hop"),
        ("5. Windowing", "Hann window per frame"),
        ("6. FFT", "512-point FFT per frame"),
        ("7. Mel filterbank", "80 triangular filters"),
        ("8. Log", "log(mel + 1e-9)"),
    ]
    for step, desc in steps:
        print(f"  {step:<25} {desc}")

    # --- 3b. CTC decoding (Connectionist Temporal Classification) ---
    print(f"\n--- 3b. CTC Decoding ---")
    print(f"  CTC allows variable-length alignment between audio and text.")
    print(f"  Formula: P(y|x) = sum over all alignments a: P(a|x)")
    print(f"  Alignments map repeated tokens to single characters.\n")

    # Simulate CTC logits over 8 time steps, 5 classes
    classes = ["<blank>", "h", "e", "l", "o"]
    random.seed(42)
    logits = [[random.gauss(0, 1) for _ in classes] for _ in range(8)]
    # Make "h-e-l-l-o" more likely
    targets = [1, 2, 3, 3, 4]  # h, e, l, l, o
    for t, idx in enumerate(targets[:5]):
        logits[t][idx] += 3.0
    # Add blank between
    logits_raw = list(logits)

    # Greedy CTC decode
    def ctc_greedy_decode(logits, classes):
        result = []
        prev = None
        for t in range(len(logits)):
            probs = [math.exp(l) for l in logits[t]]
            total = sum(probs)
            probs = [p / total for p in probs]
            best = probs.index(max(probs))
            if best != 0 and best != prev:  # skip blank and repeats
                result.append(classes[best])
            prev = best
        return "".join(result)

    decoded = ctc_greedy_decode(logits_raw, classes)
    print(f"  Greedy CTC decoded: '{decoded}'")
    print(f"  CTC blanks absorb silence and alignment gaps.")

    # --- 3c. Attention decoding ---
    print(f"\n--- 3c. Attention Decoding (Whisper-style) ---")
    print(f"  Decoder generates one token at a time, attending to encoder states.")
    print(f"  At each step t: token_t = argmax Decoder(Encoder(audio), tokens_<t)")

    target_text = "hello world"
    generated = []
    for t in range(len(target_text)):
        generated.append(target_text[t])
        context = "".join(generated)
        print(f"  Step {t:2d}: generate '{target_text[t]}' | context: '{context}'")

    # --- 3d. CTC vs Attention comparison ---
    print(f"\n--- 3d. CTC vs Attention comparison ---")
    print(f"  {'Feature':<25} {'CTC':<20} {'Attention':<20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    comparisons = [
        ("Alignment", "Independent (blank)", "Learned (cross-attn)"),
        ("Dependencies", "Conditional ind.", "Full autoregressive"),
        ("Speed", "Fast (parallel)", "Slow (sequential)"),
        ("Quality", "Good (base)", "Better (SOTA)"),
        ("External LM", "Often needed", "Built into decoder"),
    ]
    for feat, ctc, attn in comparisons:
        print(f"  {feat:<25} {ctc:<20} {attn:<20}")


# ---------------------------------------------------------------------------
# 4. Text-to-Speech — autoregressive TTS, voice cloning
# ---------------------------------------------------------------------------
def demo_text_to_speech():
    """Generate speech from text using autoregressive and neural approaches."""
    print("\n" + "=" * 70)
    print("DEMO 4: Text-to-Speech — autoregressive TTS, voice cloning concepts")
    print("=" * 70)

    # --- 4a. Text preprocessing for TTS ---
    text = "Hello, how are you today?"
    print(f"\n--- 4a. Text preprocessing for TTS ---")
    print(f"  Input: '{text}'")
    # Simple phoneme-like tokenization
    phonemes = list("".join(c if c.isalpha() else " " for c in text.lower()))
    phonemes_clean = [p for p in phonemes if p.strip()]
    print(f"  Characters: {phonemes_clean}")
    print(f"  Total: {len(phonemes_clean)} phoneme tokens")
    print(f"  In real TTS: grapheme -> phoneme -> prosody prediction")

    # --- 4b. Autoregressive mel generation ---
    print(f"\n--- 4b. Autoregressive mel spectrogram generation ---")
    print(f"  Model generates one mel frame at a time conditioned on:")
    print(f"  - Previous mel frames (autoregressive)")
    print(f"  - Text encoder output (cross-attention)")
    print(f"  - Speaker embedding (voice identity)\n")

    n_mel_frames = 8
    mel_dim = 4
    random.seed(42)
    mel_frames = []
    for t in range(n_mel_frames):
        frame = [random.gauss(0, 1) for _ in range(mel_dim)]
        mel_frames.append(frame)
        vals = [f'{v:+.3f}' for v in frame]
        print(f"  Frame {t}: [{', '.join(vals)}]")

    # --- 4c. Vocoder: mel -> waveform ---
    print(f"\n--- 4c. Vocoder (mel -> waveform) ---")
    sample_rate = 22050
    hop_length = 256
    waveform_len = n_mel_frames * hop_length
    print(f"  Mel frames: {n_mel_frames}, hop_length: {hop_length}")
    print(f"  Output waveform: {waveform_len} samples at {sample_rate} Hz")
    print(f"  Duration: {waveform_len/sample_rate:.3f} seconds")
    print(f"  Vocoder types: WaveNet, WaveRNN, HiFi-GAN, BigVGAN")

    # --- 4d. Voice cloning: speaker embedding ---
    print(f"\n--- 4d. Voice cloning concepts ---")
    # Simulate speaker embeddings for different voices
    voices = ["speaker_A", "speaker_B", "speaker_C"]
    embeddings = {}
    for voice in voices:
        random.seed(hashlib.md5(voice.encode()).hexdigest()[:8], version=1)
        embeddings[voice] = [random.gauss(0, 1) for _ in range(4)]

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x**2 for x in a)) or 1e-8
        nb = math.sqrt(sum(x**2 for x in b)) or 1e-8
        return dot / (na * nb)

    print(f"  Speaker embeddings (dim=4):")
    for voice, emb in embeddings.items():
        print(f"    {voice}: [{', '.join(f'{v:+.3f}' for v in emb)}]")

    print(f"\n  Similarity matrix:")
    print(f"  {'':>12}", end="")
    for v in voices:
        print(f"  {v:>12}", end="")
    print()
    for v1 in voices:
        print(f"  {v1:>12}", end="")
        for v2 in voices:
            sim = cosine_sim(embeddings[v1], embeddings[v2])
            print(f"  {sim:12.4f}", end="")
        print()

    print(f"\n  Voice cloning pipeline:")
    print(f"  1. Extract speaker embedding from reference audio (3-10 seconds)")
    print(f"  2. Condition TTS decoder on this embedding")
    print(f"  3. Generate mel spectrogram with cloned voice identity")
    print(f"  4. Vocoder converts mel to waveform")
    print(f"  Key models: YourTTS, OpenVoice, XTTS, Bark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_audio_tokenization()
    demo_whisper_architecture()
    demo_speech_to_text()
    demo_text_to_speech()
    print("\n" + "=" * 70)
    print("All 4 demos completed — 131-audio_speech_models.py")
    print("=" * 70)
