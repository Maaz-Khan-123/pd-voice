"""
audio_tools.py: ffmpeg conversion and the audio degradations.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

The robustness test (step 3) and the app's "degrade" slider both use these
functions, so the app shows the same kind of damage the test measured.
Every degradation takes a parselmouth.Sound and returns a new parselmouth.Sound.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------
def run_ffmpeg(args):
    """Run ffmpeg quietly; raise a readable error if it fails."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")


def convert_to_wav(in_path, out_path):
    """Convert any audio file ffmpeg understands (webm, m4a, mp3, ogg, wav...) to mono 16-bit WAV.

    The sample rate is kept as-is; extract_features() resamples to 8 kHz itself,
    so that step happens in exactly one place.
    """
    run_ffmpeg(["-i", str(in_path), "-ac", "1", "-c:a", "pcm_s16le", str(out_path)])
    return Path(out_path)


def _sound_to_wav(snd: parselmouth.Sound, path):
    sf.write(str(path), snd.values[0], int(snd.sampling_frequency), subtype="FLOAT")


def _wav_to_sound(path) -> parselmouth.Sound:
    x, sr = sf.read(str(path), always_2d=True)
    return parselmouth.Sound(x.mean(axis=1), sampling_frequency=sr)


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------
def add_noise(snd: parselmouth.Sound, snr_db: float, seed: int = 0) -> parselmouth.Sound:
    """Add white Gaussian noise at a given signal-to-noise ratio (dB).

    SNR = 10*log10(signal power / noise power). 30 dB is a quiet room, 10 dB is
    clearly noisy, 0 dB means the noise is as loud as the voice.
    A fixed seed makes the result repeatable.
    """
    x = snd.values[0]
    signal_power = np.mean(x ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.default_rng(seed).normal(0.0, np.sqrt(noise_power), size=x.shape)
    return parselmouth.Sound(x + noise, sampling_frequency=snd.sampling_frequency)


def resample_4k(snd: parselmouth.Sound) -> parselmouth.Sound:
    """Resample to 4 kHz, which throws away everything above 2 kHz.

    The training audio is already 8 kHz (0-4 kHz content), so this halves the
    bandwidth again. extract_features() later resamples back up to 8 kHz, but the
    lost high frequencies don't come back.
    """
    return snd.resample(4000)


def mp3_compress(snd: parselmouth.Sound, kbps: int) -> parselmouth.Sound:
    """Encode to MP3 at `kbps` with ffmpeg/LAME, then decode back to a waveform.

    At an 8 kHz sample rate LAME allows 8-64 kbps; lower = more compression damage.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav_in, mp3, wav_out = (Path(tmp) / n for n in ("in.wav", "x.mp3", "out.wav"))
        _sound_to_wav(snd, wav_in)
        run_ffmpeg(["-i", str(wav_in), "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(mp3)])
        run_ffmpeg(["-i", str(mp3), "-ac", "1", str(wav_out)])
        return _wav_to_sound(wav_out)


# The conditions tested in step 3: (name, family, level-for-plotting, function)
CONDITIONS = [
    ("clean", "Clean", None, lambda s, seed: s),
    ("noise_30dB", "Added noise", "30 dB SNR", lambda s, seed: add_noise(s, 30, seed)),
    ("noise_20dB", "Added noise", "20 dB SNR", lambda s, seed: add_noise(s, 20, seed)),
    ("noise_10dB", "Added noise", "10 dB SNR", lambda s, seed: add_noise(s, 10, seed)),
    ("noise_5dB", "Added noise", "5 dB SNR", lambda s, seed: add_noise(s, 5, seed)),
    ("noise_0dB", "Added noise", "0 dB SNR", lambda s, seed: add_noise(s, 0, seed)),
    ("mp3_32k", "MP3", "32 kbps", lambda s, seed: mp3_compress(s, 32)),
    ("mp3_16k", "MP3", "16 kbps", lambda s, seed: mp3_compress(s, 16)),
    ("mp3_8k", "MP3", "8 kbps", lambda s, seed: mp3_compress(s, 8)),
    ("resample_4k", "4 kHz resample", "4 kHz", lambda s, seed: resample_4k(s)),
]
