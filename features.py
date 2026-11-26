"""
features.py: the ONE feature-extraction function used everywhere in this project.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

The training recordings, the degraded copies in the robustness test, and live
recordings in the Streamlit app all go through `extract_features()`, so the model
always sees numbers computed in exactly the same way.

Pipeline for every recording:
  1. Load the audio and mix it down to mono.
  2. Resample to 8 kHz. The training data are telephone recordings at 8 kHz, so
     a 48 kHz laptop recording must be brought to the same bandwidth first.
     Otherwise its MFCCs would describe frequencies the model never saw.
  3. Peak-normalise to 1.0. Every training file was already normalised this way.
  4. Trim leading/trailing silence: keep only the span from the first to the
     last voiced (pitched) frame.
  5. Measure, with Praat (via Parselmouth):
       - mean and standard deviation of pitch (F0), in Hz
       - local jitter  (cycle-to-cycle variation in period length)
       - local shimmer (cycle-to-cycle variation in amplitude)
       - HNR, harmonics-to-noise ratio, in dB
       - the mean of each of 13 MFCCs (c0..c12) over the trimmed recording
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

# ---------------------------------------------------------------------------
# Settings (kept in one place so training and the app can't drift apart)
# ---------------------------------------------------------------------------
TARGET_SR = 8000          # Hz, matches the training data
PITCH_FLOOR = 75.0        # Hz, Praat's usual floor for adult voices
PITCH_CEILING = 500.0     # Hz, high enough for female voices
N_MFCC = 13               # c0 (overall energy) + 12 cepstral coefficients

# Column order of the feature vector. Models are trained on exactly these columns.
VOICE_FEATURES = ["f0_mean_hz", "f0_std_hz", "jitter_local", "shimmer_local", "hnr_db"]
MFCC_FEATURES = [f"mfcc{i}_mean" for i in range(N_MFCC)]
FEATURE_COLUMNS = VOICE_FEATURES + MFCC_FEATURES

# Quality-control values. They are saved but NEVER used as model inputs.
QC_COLUMNS = ["duration_s", "voiced_duration_s", "voiced_fraction"]


def load_sound(source) -> parselmouth.Sound:
    """Load a file path (or accept an existing Sound) and apply steps 1-3."""
    snd = source if isinstance(source, parselmouth.Sound) else parselmouth.Sound(str(source))
    snd = snd.convert_to_mono()
    if snd.sampling_frequency != TARGET_SR:
        # Praat's resample includes an anti-aliasing low-pass filter.
        snd = snd.resample(TARGET_SR)
    peak = np.abs(snd.values).max()
    if peak > 0:
        snd = parselmouth.Sound(snd.values / peak, sampling_frequency=TARGET_SR)
    return snd


def trim_to_voiced(snd: parselmouth.Sound):
    """Step 4: cut the sound to the span between the first and last voiced frame.

    Returns (trimmed_sound, pitch_object_of_original). If no voicing is found the
    sound is returned unchanged (the features will mostly be NaN in that case).
    """
    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]           # 0 means "unvoiced frame"
    voiced = np.where(f0 > 0)[0]
    if len(voiced) == 0:
        return snd, pitch
    t_start = pitch.xs()[voiced[0]] - pitch.time_step
    t_end = pitch.xs()[voiced[-1]] + pitch.time_step
    t_start, t_end = max(t_start, snd.xmin), min(t_end, snd.xmax)
    return snd.extract_part(from_time=t_start, to_time=t_end, preserve_times=False), pitch


def extract_features(source) -> dict:
    """Compute the feature vector for one recording.

    `source` can be a file path or a parselmouth.Sound. Returns a dict with every
    name in FEATURE_COLUMNS and QC_COLUMNS. Values that Praat cannot measure
    (e.g. no voicing found) come back as NaN rather than crashing.
    """
    snd = load_sound(source)
    duration = snd.duration
    snd, _ = trim_to_voiced(snd)

    out = {}

    # --- Pitch (F0) ---------------------------------------------------------
    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    f0_voiced = f0[f0 > 0]
    out["f0_mean_hz"] = float(np.mean(f0_voiced)) if len(f0_voiced) else np.nan
    out["f0_std_hz"] = float(np.std(f0_voiced)) if len(f0_voiced) > 1 else np.nan

    # --- Jitter and shimmer -------------------------------------------------
    # A PointProcess marks each glottal pulse (one per vocal-fold cycle).
    # The numeric arguments are Praat's standard defaults:
    #   time range 0-0 (= whole sound), shortest/longest period 0.1 ms / 20 ms,
    #   maximum period factor 1.3, maximum amplitude factor 1.6.
    try:
        pulses = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        out["jitter_local"] = call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer_local"] = call([snd, pulses], "Get shimmer (local)",
                                    0, 0, 0.0001, 0.02, 1.3, 1.6)
    except parselmouth.PraatError:
        out["jitter_local"] = out["shimmer_local"] = np.nan

    # --- Harmonics-to-noise ratio -------------------------------------------
    # Praat marks frames where HNR is undefined with -200 dB; we drop those.
    hnr = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=PITCH_FLOOR,
                                silence_threshold=0.1, periods_per_window=1.0)
    hnr_vals = hnr.values[hnr.values != -200]
    out["hnr_db"] = float(np.mean(hnr_vals)) if len(hnr_vals) else np.nan

    # --- MFCCs ----------------------------------------------------------------
    # Praat's default mel filterbank: 15 ms windows every 5 ms, filters every
    # 100 mel up to Nyquist (4 kHz here). Row 0 is c0 (energy), rows 1-12 are c1-c12.
    mfcc = snd.to_mfcc(number_of_coefficients=N_MFCC - 1).to_array()
    for i in range(N_MFCC):
        out[f"mfcc{i}_mean"] = float(np.mean(mfcc[i]))

    # --- Quality-control numbers (not model inputs) ---------------------------
    out["duration_s"] = float(duration)
    out["voiced_duration_s"] = float(len(f0_voiced) * pitch.time_step)
    out["voiced_fraction"] = float(len(f0_voiced) / max(len(f0), 1))

    # Praat returns undefined measurements as NaN already; make sure everything is a float.
    return {k: float(v) if v is not None else np.nan for k, v in out.items()}


# ---------------------------------------------------------------------------
# Sanity-check features: properties of the RECORDING, not of the voice.
# ---------------------------------------------------------------------------
# If a classifier can separate the groups using only these, the dataset has
# recording-setup differences between groups, and any "voice" result must be
# read with suspicion. These are never fed to the voice model.
RECORDING_COLUMNS = ["rec_duration_s", "rec_loudness_dbfs", "rec_noise_floor_dbfs",
                     "rec_lead_silence_s", "rec_trail_silence_s"]

FRAME_S = 0.025           # 25 ms analysis frames
HOP_S = 0.010             # every 10 ms
SILENCE_DB_BELOW_MAX = 30 # a frame is "silent" if it is >30 dB quieter than the loudest frame


def recording_features(source) -> dict:
    """Duration, loudness, noise floor and leading/trailing silence of the raw file.

    Computed on the file as stored (no resampling, normalisation or trimming), so
    that it reflects how the recording was made and processed.
    """
    snd = source if isinstance(source, parselmouth.Sound) else parselmouth.Sound(str(source))
    x = snd.convert_to_mono().values[0]
    sr = snd.sampling_frequency
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)

    # Frame-by-frame RMS energy, in dB relative to full scale (dBFS).
    starts = range(0, max(len(x) - frame, 0) + 1, hop)
    rms = np.array([np.sqrt(np.mean(x[s:s + frame] ** 2)) for s in starts])
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10))

    loud = np.where(rms_db > rms_db.max() - SILENCE_DB_BELOW_MAX)[0]
    return {
        "rec_duration_s": len(x) / sr,
        "rec_loudness_dbfs": float(20 * np.log10(max(np.sqrt(np.mean(x ** 2)), 1e-10))),
        # 5th percentile of frame energy = how loud the quietest parts are
        "rec_noise_floor_dbfs": float(np.percentile(rms_db, 5)),
        "rec_lead_silence_s": float(loud[0] * hop / sr),
        "rec_trail_silence_s": float((len(rms_db) - 1 - loud[-1]) * hop / sr),
    }


def analysis_traces(source) -> dict:
    """Extra data the Streamlit app needs for its plots (waveform, pitch, spectrogram).

    Uses the same loading steps as extract_features so the plots show exactly
    what the model analysed.
    """
    snd = load_sound(source)
    pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"].copy()
    f0[f0 == 0] = np.nan                                  # gaps where unvoiced
    spec = snd.to_spectrogram(window_length=0.03, maximum_frequency=TARGET_SR / 2)
    return {
        "time": snd.xs(), "samples": snd.values[0],
        "pitch_time": pitch.xs(), "pitch_hz": f0,
        "spec_time": spec.xs(), "spec_freq": spec.ys(),
        "spec_db": 10 * np.log10(np.maximum(spec.values, 1e-12)),
    }
