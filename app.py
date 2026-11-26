"""
app.py: Streamlit front end for the voice research prototype.

RESEARCH PROTOTYPE, NOT A DIAGNOSIS.

Run:  .venv/bin/streamlit run app.py

Layout
  Header + permanent "Research prototype, not a diagnosis" banner (also pinned to
  the bottom of the window).
  Tab 1, Record / Upload: get audio, check it, show waveform / pitch / spectrogram.
  Tab 2, Your Voice vs. the Dataset: score range + every feature against the
         81 training speakers.
  Tab 3, How Noise Changes the Result: degradation slider with live before/after.
  Footer: "How this works and what it can't tell you".

Every number shown comes from the same extract_features() used in training
(features.py) and the same degradations used in the robustness test (audio_tools.py).
"""

import io
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import parselmouth
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st
import streamlit.components.v2  # noqa: F401  (raw-microphone component below)
from plotly.subplots import make_subplots

from features import (extract_features, analysis_traces, load_sound, FEATURE_COLUMNS,
                      VOICE_FEATURES, MFCC_FEATURES, TARGET_SR)
from audio_tools import convert_to_wav, add_noise, mp3_compress, resample_4k, level_drop_db
from common import make_model, grouped_folds

# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
# One accent (aqua) for "your recording" and UI; blue = healthy group,
# orange = Parkinson's group. Validated together for colour-blind separation.
ACCENT, HC_COLOR, PD_COLOR = "#00a888", "#338ced", "#e76624"
BG, PANEL, GRID = "#131315", "#1c1c20", "#2a2a30"
TEXT, TEXT_2 = "#ececec", "#9c9c9c"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

PROTOTYPE = "Research prototype, not a diagnosis"

# Permutation-test p-values from step2_train_evaluate.py (results/step2_log.txt).
# Update these if the analysis is re-run with different data.
PERM_P_VOICE, PERM_P_RECORDING = 0.19, 0.002

ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac", ".aac", ".mp4", ".opus"}
MIN_VOICED_S = 1.0        # below this we refuse to score
WARN_VOICED_S = 2.0       # below this we score but warn
MAX_ANALYSE_S = 30.0      # longer recordings are cut to the first 30 s
SILENCE_PEAK = 0.003      # peak amplitude below ~-50 dBFS = silence
LEVEL_DROP_WARN_DB = 15   # volume collapse after the first 0.5 s (see level_drop_db)
# Example for demos without a microphone (a healthy-control training recording).
# (copy of figshare file AH_121A_BD5BA248-E807-4CB9-8B53-47E7FFE5F8E2.wav)
EXAMPLE_FILE = "examples/example_healthy_control.wav"

FEATURE_LABELS = {
    "f0_mean_hz": ("Mean pitch", "Hz", "How high the voice is."),
    "f0_std_hz": ("Pitch steadiness (SD)", "Hz", "How much the pitch wobbles. Lower = steadier."),
    "jitter_local": ("Jitter", "ratio", "Cycle-to-cycle timing variation of the vocal folds."),
    "shimmer_local": ("Shimmer", "ratio", "Cycle-to-cycle loudness variation."),
    "hnr_db": ("Harmonics-to-noise ratio", "dB", "Clear tone vs breathy noise. Higher = clearer."),
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* page width and spacing */
.block-container {{ max-width: 1120px; padding-top: 2.2rem; padding-bottom: 6rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* header */
.rp-header {{ display: flex; align-items: flex-end; justify-content: space-between;
             gap: 1rem; flex-wrap: wrap; margin-bottom: 0.9rem; }}
.rp-title {{ font-size: 2rem !important; font-weight: 700; line-height: 1.15; margin: 0; }}
.rp-sub {{ color: {TEXT_2}; font-size: 0.95rem; margin-top: 0.3rem; }}
.rp-banner {{ display: flex; gap: 0.75rem; align-items: center; padding: 0.7rem 1rem;
             border: 1px solid #5c3a24; background: #2a1d15; border-radius: 0.6rem;
             color: #f6c9ad; font-size: 0.92rem; margin-bottom: 1.2rem; }}
.rp-banner b {{ color: #ffd9c2; }}

/* permanent ribbon pinned to the bottom of the window */
.rp-ribbon {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000;
             text-align: center; padding: 0.45rem 1rem calc(0.45rem + env(safe-area-inset-bottom, 0px));
             background: #2a1d15; border-top: 1px solid #5c3a24; color: #f6c9ad;
             font-size: 0.82rem; font-weight: 600; letter-spacing: 0.02em; }}

/* cards and stats */
.rp-card {{ background: {PANEL}; border: 1px solid {GRID}; border-radius: 0.7rem;
           padding: 1rem 1.1rem; height: 100%; }}
.rp-card h4 {{ margin: 0 0 0.35rem 0; font-size: 0.95rem; font-weight: 600; }}
.rp-card p {{ margin: 0; color: {TEXT_2}; font-size: 0.88rem; line-height: 1.45; }}
.rp-stat {{ font-size: 1.6rem; font-weight: 700; line-height: 1.1; }}
.rp-stat-label {{ color: {TEXT_2}; font-size: 0.8rem; margin-top: 0.15rem; }}
.rp-range {{ font-size: 2.1rem; font-weight: 700; color: {ACCENT}; line-height: 1.1; }}
.rp-muted {{ color: {TEXT_2}; font-size: 0.86rem; }}

/* quality-check chips */
.rp-chips {{ display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.4rem 0 0.8rem 0; }}
.rp-chip {{ border-radius: 999px; padding: 0.22rem 0.7rem; font-size: 0.8rem;
           border: 1px solid {GRID}; background: {PANEL}; }}
.rp-chip.ok {{ border-color: #145c4e; color: #7fe0c8; }}
.rp-chip.warn {{ border-color: #6b5214; color: #f5d27a; }}

/* tabs: larger, evenly spaced */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {GRID}; }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.98rem; font-weight: 600; padding: 0.6rem 0.1rem; }}

/* softer audio player */
[data-testid="stAudio"] {{ margin: 0.2rem 0 0.6rem 0; }}
</style>
"""


# ---------------------------------------------------------------------------
# Errors that should become a friendly message, not a crash
# ---------------------------------------------------------------------------
class AudioProblem(Exception):
    """A problem with the user's audio. `title` is short; `advice` says how to fix it."""

    def __init__(self, title: str, advice: str):
        super().__init__(title)
        self.title, self.advice = title, advice


# ---------------------------------------------------------------------------
# Training artefacts (loaded once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    return joblib.load("models/voice_models.joblib")


@st.cache_data
def load_training_features() -> pd.DataFrame:
    return pd.read_csv("results/features.csv")


@st.cache_data
def held_out_scores(model_name: str) -> pd.DataFrame:
    """Score every training speaker with a model that did NOT see them (grouped 5-fold).

    Shown next to the user's score so they can see how much the two groups overlap.
    """
    train = load_training_features()
    X, y = train[FEATURE_COLUMNS].to_numpy(float), train["label"].to_numpy()
    groups = train["sample_id"].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in grouped_folds(y, groups, 0):
        oof[te] = make_model(model_name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pd.DataFrame({"label": y, "score": oof})


def score_range(feats: dict, model_name: str):
    """(central, 5th pct, 95th pct): full model + spread over 100 bootstrap models."""
    bundle = load_models()
    x = np.array([[feats[c] for c in bundle["feature_columns"]]], dtype=float)
    if np.isnan(x).any():
        return None
    m = bundle["models"][model_name]
    point = m["full"].predict_proba(x)[0, 1]
    boot = np.array([b.predict_proba(x)[0, 1] for b in m["bootstrap"]])
    return float(point), float(np.percentile(boot, 5)), float(np.percentile(boot, 95))


# ---------------------------------------------------------------------------
# Audio input, conversion and checks
# ---------------------------------------------------------------------------
def wav_bytes_to_sound(wav: bytes) -> parselmouth.Sound:
    x, sr = sf.read(io.BytesIO(wav), always_2d=True)
    return parselmouth.Sound(x.mean(axis=1), sampling_frequency=sr)


def sound_to_wav_bytes(snd: parselmouth.Sound) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(snd.values[0], -1, 1), int(snd.sampling_frequency), format="WAV")
    return buf.getvalue()


# Visitors' audio is only ever held in memory: at most 64 items, dropped after 1 hour.
@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def prepare_audio(raw: bytes, filename: str) -> tuple[bytes, list[str]]:
    """Validate and convert any upload/recording to WAV with ffmpeg.

    Returns (wav_bytes, notes). Raises AudioProblem with a plain-language message
    for empty, unsupported, unreadable, silent or too-short audio.
    """
    notes = []
    ext = Path(filename).suffix.lower()
    if not raw:
        raise AudioProblem("The file is empty.", "Try recording again, or choose another file.")
    if ext not in ALLOWED_EXT:
        raise AudioProblem(
            f"“{ext or 'no extension'}” isn't a supported audio type.",
            "Please use WAV, MP3, M4A, OGG, WEBM, FLAC, AAC or OPUS.")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"input{ext}"
        src.write_bytes(raw)
        try:
            wav_path = convert_to_wav(src, Path(tmp) / "converted.wav")
        except (RuntimeError, FileNotFoundError):
            raise AudioProblem(
                "This file couldn't be read as audio.",
                "It may be damaged, or it may not really be an audio file. Try exporting it "
                "again as WAV or MP3.")
        wav = wav_path.read_bytes()

    try:
        snd = wav_bytes_to_sound(wav)
    except Exception:
        raise AudioProblem("This file couldn't be read as audio.",
                           "Try exporting it again as WAV or MP3.")

    if snd.duration < 0.5:
        raise AudioProblem(f"The recording is only {snd.duration:.1f} s long.",
                           "Hold a steady “ahhh” for 3-5 seconds.")
    if np.abs(snd.values).max() < SILENCE_PEAK:
        raise AudioProblem(
            "We only heard silence.",
            "Check that the browser has microphone permission and the right input device "
            "is selected, then try again a little closer to the microphone.")
    if snd.duration > MAX_ANALYSE_S:
        snd = snd.extract_part(0, MAX_ANALYSE_S, preserve_times=False)
        wav = sound_to_wav_bytes(snd)
        notes.append(f"The recording was longer than {MAX_ANALYSE_S:.0f} s, so only the first "
                     f"{MAX_ANALYSE_S:.0f} s were analysed.")
    clip = float(np.mean(np.abs(snd.values) >= 0.99))
    if clip > 0.01:
        notes.append(f"{clip:.0%} of the recording is clipped (too loud). Moving a little "
                     "further from the microphone would give cleaner measurements.")
    return wav, notes


DEGRADATIONS = {
    "Background noise": {"levels": ["None", "40 dB", "30 dB", "20 dB", "10 dB", "5 dB", "0 dB"],
                         "start": 3,
                         "help": "Signal-to-noise ratio. 30 dB ≈ quiet room, 10 dB ≈ busy café, "
                                 "0 dB = noise as loud as the voice."},
    "MP3 compression": {"levels": ["None", "64 kbps", "32 kbps", "16 kbps", "8 kbps"],
                        "start": 3,
                        "help": "Lower bitrate = more compression damage, like a bad call."},
    "Low bandwidth": {"levels": ["None", "4 kHz"],
                      "start": 1,
                      "help": "Resample to 4 kHz: removes everything above 2 kHz."},
}


def apply_degradation(snd: parselmouth.Sound, kind: str, level: str) -> parselmouth.Sound:
    """Same degradation functions as the step 3 robustness test."""
    if level == "None":
        return snd
    if kind == "Background noise":
        return add_noise(snd, float(level.split()[0]), seed=0)
    if kind == "MP3 compression":
        return mp3_compress(snd, int(level.split()[0]))
    return resample_4k(snd)


@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def analyse(wav: bytes, kind: str, level: str, model_name: str) -> dict:
    """Degrade (optionally) → extract features → score → plot traces. Cached per input."""
    snd = load_sound(wav_bytes_to_sound(wav))          # mono, 8 kHz, peak-normalised
    snd = apply_degradation(snd, kind, level)
    feats = extract_features(snd)
    traces = analysis_traces(snd)
    return {"feats": feats, "score": score_range(feats, model_name),
            "traces": traces, "wav": sound_to_wav_bytes(load_sound(snd)),
            "level_drop_db": level_drop_db(snd)}


def check_voicing(feats: dict, level_drop: float = 0.0):
    """Refuse to score recordings with too little voice; return warnings otherwise."""
    voiced = feats.get("voiced_duration_s", 0) or 0
    if np.isnan(feats["f0_mean_hz"]) or voiced == 0:
        raise AudioProblem(
            "No voice pitch was detected.",
            "This can happen with silence, whispering, or only background noise. Say a "
            "steady, voiced “ahhh” at a comfortable pitch.")
    if voiced < MIN_VOICED_S:
        raise AudioProblem(
            f"Only {voiced:.1f} s of steady voice was found. That's too short to measure.",
            "Hold the “ahhh” for 3-5 seconds without stopping.")
    warnings = []
    if voiced < WARN_VOICED_S:
        warnings.append(f"Only {voiced:.1f} s of steady voice. Jitter and shimmer are less "
                        "reliable on short samples; 3-5 s is better.")
    if level_drop > LEVEL_DROP_WARN_DB:
        warnings.append(
            f"The volume dropped by about {level_drop:.0f} dB after the first half-second. If "
            "you didn't get quieter on purpose, your device is probably applying noise "
            "suppression (it treats a steady “ahhh” as background noise). Try turning off "
            "“Voice Isolation” / noise cancelling (macOS: Control Centre → Mic Mode → "
            "Standard), or record with another app and upload the file.")
    if feats["voiced_fraction"] < 0.7:
        warnings.append("The pitch dropped out for part of the recording (breaks, noise "
                        "or a fading voice).")
    train = load_training_features()
    outside = [FEATURE_LABELS[c][0] for c in VOICE_FEATURES
               if not (train[c].min() <= feats[c] <= train[c].max())]
    if outside:
        warnings.append("Outside the range of all 81 training speakers: " + ", ".join(outside)
                        + ". The model has never seen values like this.")
    return warnings


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------
def style(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    """Shared dark styling: transparent background, recessive grid, Inter font."""
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT, size=12, color=TEXT),
        margin=dict(l=8, r=8, t=40 if title else 10, b=8), showlegend=False,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=14)) if title else None,
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font=dict(family=FONT, color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_2))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=TEXT_2))
    return fig


def show(fig: go.Figure, key: str):
    """Render a Plotly chart. `key` is required: without it Streamlit derives an ID
    from the figure's contents, and two identical figures (e.g. original vs
    "degraded" at severity None) crash with StreamlitDuplicateElementId."""
    st.plotly_chart(fig, key=key, width="stretch",
                    config={"displayModeBar": False, "responsive": True})


def _envelope(t, x, n=1200):
    """Min/max envelope so a 30 s waveform still draws quickly."""
    if len(x) <= 2 * n:
        return t, x, x
    k = len(x) // n
    xr = x[: k * n].reshape(n, k)
    return t[: k * n].reshape(n, k)[:, 0], xr.min(axis=1), xr.max(axis=1)


def _add_wave(fig, tr, color, name, opacity=1.0, row=None, col=None):
    t, lo, hi = _envelope(np.asarray(tr["time"]), np.asarray(tr["samples"]))
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=0, color=color),
                             hoverinfo="skip", name=name, opacity=opacity), **kw)
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=0.6, color=color),
                             fill="tonexty", fillcolor=_rgba(color, 0.55 * opacity),
                             name=name, hovertemplate="%{x:.2f} s<extra>" + name + "</extra>"),
                  **kw)


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a:.2f})"


def _spec_grid(tr, max_cols=400):
    """Average spectrogram frames in time so the heatmap stays light."""
    db = np.asarray(tr["spec_db"])
    t = np.asarray(tr["spec_time"])
    k = max(1, db.shape[1] // max_cols)
    n = db.shape[1] // k
    return t[: n * k].reshape(n, k).mean(1), np.asarray(tr["spec_freq"]), \
        db[:, : n * k].reshape(db.shape[0], n, k).mean(2)


# Single-hue sequential scale: background → accent → near-white.
SPEC_SCALE = [[0, BG], [0.55, "#0d5c4d"], [0.8, ACCENT], [1, "#d9fff5"]]


def fig_signal(tr) -> go.Figure:
    """Waveform, pitch trace and spectrogram, stacked with a shared time axis."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.28, 0.27, 0.45],
                        subplot_titles=("Waveform", "Pitch (gaps = no pitch detected)",
                                        "Spectrogram, 0-4 kHz (brighter = more energy)"))
    _add_wave(fig, tr, ACCENT, "Waveform", row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["pitch_time"], y=tr["pitch_hz"], mode="lines",
                             line=dict(color=ACCENT, width=2), connectgaps=False,
                             hovertemplate="%{x:.2f} s · %{y:.0f} Hz<extra></extra>"), row=2, col=1)
    t, f, db = _spec_grid(tr)
    fig.add_trace(go.Heatmap(x=t, y=f, z=db, colorscale=SPEC_SCALE, zmax=db.max(),
                             zmin=db.max() - 70, showscale=False,
                             hovertemplate="%{x:.2f} s · %{y:.0f} Hz · %{z:.0f} dB<extra></extra>"),
                  row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, range=[-1.05, 1.05])
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    fig.update_yaxes(title_text="Hz", row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    style(fig, 620)
    fig.update_annotations(font=dict(size=12, color=TEXT_2), x=0, xanchor="left")
    return fig


def fig_score(score, oof: pd.DataFrame) -> go.Figure:
    """Your score range on the same 0-1 axis as the 81 held-out dataset speakers."""
    point, lo, hi = score
    rng = np.random.default_rng(1)
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=1, fillcolor=PD_COLOR, opacity=0.05, line_width=0)
    fig.add_vrect(x0=0, x1=0.5, fillcolor=HC_COLOR, opacity=0.05, line_width=0)
    for lab, name, color, y in ((0, "Healthy speakers (held out)", HC_COLOR, 2),
                                (1, "Parkinson's speakers (held out)", PD_COLOR, 1)):
        s = oof.loc[oof.label == lab, "score"]
        fig.add_trace(go.Scatter(
            x=s, y=y + rng.uniform(-0.22, 0.22, len(s)), mode="markers", name=name,
            marker=dict(size=8, color=color, line=dict(width=1, color=BG)),
            hovertemplate=f"{name}<br>score %{{x:.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0, 0], mode="lines",
                             line=dict(color=ACCENT, width=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[point], y=[0], mode="markers",
        marker=dict(size=18, color=ACCENT, symbol="diamond", line=dict(width=2, color=TEXT)),
        hovertemplate=f"Your recording<br>range {lo:.2f}-{hi:.2f}<br>central %{{x:.2f}}<extra></extra>"))
    fig.add_vline(x=0.5, line=dict(color=TEXT_2, dash="dash", width=1))
    fig.update_yaxes(tickvals=[0, 1, 2], range=[-0.6, 2.5],
                     ticktext=["<b>Your recording</b>", "Parkinson's (dataset)", "Healthy (dataset)"],
                     showgrid=False)
    fig.update_xaxes(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1],
                     title_text="← more like the healthy group      ·      more like the Parkinson's group →",
                     title_font=dict(size=12, color=TEXT_2))
    return style(fig, 300)


def fig_voice_features(feats: dict, train: pd.DataFrame) -> go.Figure:
    """One row per voice feature: dataset distributions (violins + dots), you = aqua diamond."""
    n = len(VOICE_FEATURES)
    titles = [f"<b>{FEATURE_LABELS[c][0]}</b> ({FEATURE_LABELS[c][1]}) · "
              f"<span style='color:{TEXT_2}'>{FEATURE_LABELS[c][2]}</span>" for c in VOICE_FEATURES]
    fig = make_subplots(rows=n, cols=1, vertical_spacing=0.085, subplot_titles=titles)
    for i, c in enumerate(VOICE_FEATURES, start=1):
        for lab, name, color in ((0, "Healthy", HC_COLOR), (1, "Parkinson's", PD_COLOR)):
            v = train.loc[train.label == lab, c]
            fig.add_trace(go.Violin(
                x=v, y=[name] * len(v), orientation="h", side="both", width=0.8,
                line=dict(color=color, width=1), fillcolor=_rgba(color, 0.18),
                points="all", pointpos=0, jitter=0.6,
                marker=dict(size=5, color=color, opacity=0.85),
                meanline=dict(visible=False), box=dict(visible=False), spanmode="hard",
                hovertemplate=f"{name}: %{{x:.4g}}<extra></extra>", name=name), row=i, col=1)
        you = feats[c]
        fig.add_vline(x=you, line=dict(color=ACCENT, width=2), row=i, col=1)
        fig.add_trace(go.Scatter(
            x=[you, you], y=["Healthy", "Parkinson's"], mode="markers",
            marker=dict(size=14, symbol="diamond", color=ACCENT, line=dict(width=1.5, color=TEXT)),
            hovertemplate=f"You: {you:.4g}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(categoryorder="array", categoryarray=["Parkinson's", "Healthy"],
                         row=i, col=1, showgrid=False)
    style(fig, 150 * n + 40)
    fig.update_layout(margin=dict(t=34))           # room for the first row's title
    fig.update_annotations(font=dict(size=12, color=TEXT), x=0, xanchor="left")
    return fig


def fig_mfcc(feats: dict, train: pd.DataFrame) -> go.Figure:
    """All 13 MFCC means as z-scores (relative to all 81 speakers) on one chart."""
    mu, sd = train[MFCC_FEATURES].mean(), train[MFCC_FEATURES].std()
    names = [f"c{i}" for i in range(len(MFCC_FEATURES))]
    fig = go.Figure()
    for lab, name, color in ((0, "Healthy", HC_COLOR), (1, "Parkinson's", PD_COLOR)):
        z = (train.loc[train.label == lab, MFCC_FEATURES] - mu) / sd
        long = z.melt(var_name="f", value_name="z")
        fig.add_trace(go.Box(
            x=long.f.map(dict(zip(MFCC_FEATURES, names))), y=long.z, name=name,
            marker=dict(color=color, size=4), line=dict(color=color, width=1),
            fillcolor=_rgba(color, 0.15), boxpoints="all", jitter=0.5, pointpos=0,
            hovertemplate=f"{name} %{{x}}: z = %{{y:.2f}}<extra></extra>"))
    you = [(feats[c] - mu[c]) / sd[c] for c in MFCC_FEATURES]
    fig.add_trace(go.Scatter(x=names, y=you, mode="markers", name="You",
                             marker=dict(size=13, symbol="diamond", color=ACCENT,
                                         line=dict(width=1.5, color=TEXT)),
                             hovertemplate="You %{x}: z = %{y:.2f}<extra></extra>"))
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title_text="Standard deviations from the dataset average")
    style(fig, 380)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0,
                                                   font=dict(color=TEXT_2)))
    return fig


def fig_wave(tr, color, title) -> go.Figure:
    """One waveform panel (used side by side for before/after)."""
    fig = go.Figure()
    _add_wave(fig, tr, color, title)
    fig.update_yaxes(range=[-1.05, 1.05], title_text="Amplitude")
    fig.update_xaxes(title_text="Time (s)")
    return style(fig, 190, title)


def fig_spec(tr, title) -> go.Figure:
    t, f, db = _spec_grid(tr, max_cols=250)
    fig = go.Figure(go.Heatmap(x=t, y=f, z=db, colorscale=SPEC_SCALE, zmax=db.max(),
                               zmin=db.max() - 70, showscale=False,
                               hovertemplate="%{x:.2f} s · %{y:.0f} Hz<extra></extra>"))
    fig.update_yaxes(title_text="Hz")
    fig.update_xaxes(title_text="Time (s)")
    return style(fig, 230, title)


def fig_curve(levels, results, current) -> go.Figure:
    xs = list(range(len(levels)))
    ok = [i for i, r in enumerate(results) if r is not None]
    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=1, fillcolor=PD_COLOR, opacity=0.05, line_width=0)
    fig.add_hrect(y0=0, y1=0.5, fillcolor=HC_COLOR, opacity=0.05, line_width=0)
    fig.add_trace(go.Scatter(x=[xs[i] for i in ok] + [xs[i] for i in ok][::-1],
                             y=[results[i][2] for i in ok] + [results[i][1] for i in ok][::-1],
                             fill="toself", fillcolor=_rgba(ACCENT, 0.2), line=dict(width=0),
                             mode="lines", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[xs[i] for i in ok], y=[results[i][0] for i in ok],
                             mode="lines+markers", line=dict(color=ACCENT, width=2),
                             marker=dict(size=8, color=ACCENT),
                             customdata=[[results[i][1], results[i][2]] for i in ok],
                             hovertemplate="score %{y:.2f} (range %{customdata[0]:.2f}-"
                                           "%{customdata[1]:.2f})<extra></extra>"))
    if results[current] is not None:
        fig.add_trace(go.Scatter(x=[current], y=[results[current][0]], mode="markers",
                                 marker=dict(size=16, symbol="diamond", color=ACCENT,
                                             line=dict(width=2, color=TEXT)), hoverinfo="skip"))
    fig.add_hline(y=0.5, line=dict(color=TEXT_2, dash="dash", width=1))
    fig.update_xaxes(tickvals=xs, ticktext=levels, title_text="Severity")
    fig.update_yaxes(range=[0, 1], title_text="Score")
    return style(fig, 260, "Score at every severity (band = range across 100 models)")


# ---------------------------------------------------------------------------
# Plain-language text
# ---------------------------------------------------------------------------
def explain_score(score, oof: pd.DataFrame) -> str:
    """Describe the range without ever giving a verdict."""
    point, lo, hi = score
    hc_high = (oof[oof.label == 0].score >= 0.5).mean()
    pd_low = (oof[oof.label == 1].score < 0.5).mean()
    width = hi - lo
    if lo < 0.5 < hi:
        msg = ("**The range crosses the middle line**, so the models don't agree on which "
               "group this recording resembles. It sits where the healthy and Parkinson's "
               "speakers in the dataset overlap.")
    elif hi <= 0.5:
        msg = (f"**Every model places this recording on the healthy-group side.** This does "
               f"not rule anything out: when held out, {pd_low:.0%} of the Parkinson's "
               f"speakers in the dataset also landed on this side.")
    else:
        msg = (f"**Every model places this recording on the Parkinson's-group side.** This "
               f"does **not** mean you have Parkinson's: when held out, {hc_high:.0%} of the "
               f"*healthy* speakers in the dataset also landed on this side, and background "
               f"noise, the microphone and age all push scores this way (see tab 3).")
    if width > 0.4:
        msg += (f" The range is wide ({width:.2f}), so the answer depends a lot on which "
                "81 speakers happened to be in the training data.")
    return msg


# ---------------------------------------------------------------------------
# Raw microphone: switch off the browser's voice-call processing
# ---------------------------------------------------------------------------
# st.audio_input asks for the mic with getUserMedia({audio: true}). Browsers then
# apply noise suppression, auto gain and echo cancellation by default. Noise
# suppression treats a steady "ahhh" as background noise and fades it out within
# 1-2 s (we measured ~18 dB on a constant test vowel in Chrome), which changes the
# MFCCs and HNR the model reads. This invisible component runs in the app page and
# wraps getUserMedia so audio requests ask for the RAW microphone instead.
# Streamlit's recorder itself is unchanged.
RAW_MIC_JS = """
export default function () {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__pdVoiceRawMic) return;   // patch once per page
  const original = md.getUserMedia.bind(md);
  md.getUserMedia = (constraints) => {
    if (constraints && constraints.audio) {
      const base = typeof constraints.audio === "object" ? constraints.audio : {};
      constraints = { ...constraints, audio: { ...base,
        noiseSuppression: false, autoGainControl: false, echoCancellation: false } };
    }
    return original(constraints).then((stream) => {
      const track = stream.getAudioTracks()[0];
      if (track) window.__pdVoiceMicSettings = track.getSettings();  // for debugging
      return stream;
    });
  };
  md.__pdVoiceRawMic = true;
}
"""
_raw_mic = st.components.v2.component("pd_voice.raw_mic", js=RAW_MIC_JS)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def header():
    _raw_mic(key="raw_mic")          # must be on the page before anyone presses record
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="rp-header">
          <div>
            <div class="rp-title">Sustained-Vowel Voice Analysis</div>
            <div class="rp-sub">What a 3-second “ahhh” does and doesn't reveal, measured with Praat</div>
          </div>
        </div>
        <div class="rp-banner">⚠️ <div><b>{PROTOTYPE}.</b> A high-school research project trained
          on 81 telephone recordings. It cannot tell you whether you have Parkinson's disease.
          If you have health concerns, talk to a doctor.</div></div>
        <div class="rp-ribbon">⚠️ {PROTOTYPE.upper()}</div>
        """, unsafe_allow_html=True)


def empty_state(where: str):
    st.markdown(f"<div class='rp-card'><h4>No recording yet</h4><p>Record or upload a "
                f"steady “ahhh” in the <b>Record / Upload</b> tab to see {where}.</p></div>",
                unsafe_allow_html=True)


def show_problem(p: AudioProblem):
    st.error(f"**{p.title}**  \n{p.advice}", icon="🎙️")


def tab_record():
    tips = st.columns(3)
    for col, (h, p) in zip(tips, [
        ("1 · Breathe in", "Sit upright in a quiet room, 20-30 cm from the microphone."),
        ("2 · Say “ahhh”", "One steady vowel at a comfortable pitch and loudness."),
        ("3 · Hold it 3-5 s", "Don't change pitch, don't stop. Then press stop."),
    ]):
        col.markdown(f"<div class='rp-card'><h4>{h}</h4><p>{p}</p></div>", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    v = st.session_state.get("input_version", 0)     # bumped by "Delete my audio"
    recorded = c1.audio_input("Record with your microphone", key=f"rec_{v}")
    c1.caption("🔒 Your audio is analysed in memory and never saved to disk or shared. "
               "Browser noise suppression and auto-volume are switched off so a steady "
               "“ahhh” isn't faded out.")
    uploaded = c2.file_uploader("…or upload an audio file", key=f"up_{v}",
                                help="WAV, MP3, M4A, OGG, WEBM, FLAC, AAC or OPUS, up to 25 MB")

    # Demo fallback: one recording from the dataset, for when there's no microphone.
    ex1, ex2 = st.columns([1, 3])
    if ex1.button("Try an example recording", disabled=bool(uploaded or recorded)):
        st.session_state.use_example = not st.session_state.get("use_example", False)
    # ?example=1 in the URL pre-loads the example (handy as a demo bookmark).
    if st.query_params.get("example") == "1" and "use_example" not in st.session_state:
        st.session_state.use_example = True
    use_example = st.session_state.get("use_example", False) and not (uploaded or recorded)
    if use_example:
        ex2.caption("Using a recording **from the training dataset**. The model has already "
                    "heard this speaker, so its score is more optimistic than for a new voice. "
                    "Click again to clear.")

    if uploaded and recorded:
        st.caption("Using the uploaded file. Remove it to analyse your recording instead.")
    if uploaded or recorded:
        src = uploaded or recorded
        return src.getvalue(), getattr(src, "name", "") or "recording.wav"
    if use_example:
        return Path(EXAMPLE_FILE).read_bytes(), Path(EXAMPLE_FILE).name
    return None


def tab_dataset(base: dict, model_name: str):
    train = load_training_features()
    oof = held_out_scores(model_name)
    st.markdown("#### Where this recording falls")
    if base["score"] is None:
        st.warning("Some features couldn't be measured, so no score can be shown.")
    else:
        point, lo, hi = base["score"]
        c1, c2 = st.columns([1, 2.2], gap="large")
        with c1:
            st.markdown(f"<div class='rp-card'><div class='rp-stat-label'>Score range</div>"
                        f"<div class='rp-range'>{lo:.2f} – {hi:.2f}</div>"
                        f"<div class='rp-muted'>central estimate {point:.2f} · scale 0-1</div>"
                        f"<div class='rp-muted' style='margin-top:.6rem'><b>{PROTOTYPE}</b>"
                        f"</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(explain_score(base["score"], oof))
            st.caption("The score measures how much this recording's 18 measurements resemble "
                       "the Parkinson's group of this small dataset. It is **not** a probability "
                       "of having Parkinson's. The range is the 5th-95th percentile over 100 "
                       "models, each trained on a resampled copy of the data.")
        show(fig_score(base["score"], oof), key="chart_score")
        st.caption("Dots: each of the 81 dataset speakers, scored by a model that never saw "
                   "them. The heavy overlap is why no single score can be trusted.")

    st.markdown("#### Your voice measurements vs. the dataset")
    st.caption("Violins and dots: the 41 healthy and 40 Parkinson's speakers. Aqua diamond: "
               "this recording. Hover for exact values.")
    show(fig_voice_features(base["feats"], train), key="chart_voice_features")
    with st.expander("Timbre: 13 MFCC means (spectral shape)"):
        st.caption("MFCCs summarise the shape of the sound spectrum. Shown as distance from the "
                   "dataset average, in standard deviations, so all 13 fit on one axis.")
        show(fig_mfcc(base["feats"], train), key="chart_mfcc")
    with st.expander("All 18 numbers as a table"):
        f = base["feats"]
        tbl = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "this recording": [f[c] for c in FEATURE_COLUMNS],
            "healthy median": [train.loc[train.label == 0, c].median() for c in FEATURE_COLUMNS],
            "Parkinson's median": [train.loc[train.label == 1, c].median() for c in FEATURE_COLUMNS],
            "% healthy below you": [(train.loc[train.label == 0, c] < f[c]).mean() * 100
                                    for c in FEATURE_COLUMNS],
            "% Parkinson's below you": [(train.loc[train.label == 1, c] < f[c]).mean() * 100
                                        for c in FEATURE_COLUMNS],
        })
        st.dataframe(tbl.round(3), hide_index=True, width="stretch")


def tab_noise(wav: bytes, base: dict, model_name: str):
    st.markdown("#### Same voice, worse recording")
    st.caption("Nothing about the voice changes here, only the recording quality. If the "
               "score moves, it's reacting to the recording, not to health.")
    kind = st.segmented_control("Degradation", list(DEGRADATIONS), default="Background noise",
                                key="deg_kind") or "Background noise"
    levels = DEGRADATIONS[kind]["levels"]
    # Open on a level that visibly changes something, so the effect is clear at once.
    level = st.select_slider("Severity", options=levels, key=f"deg_{kind}",
                             value=levels[DEGRADATIONS[kind]["start"]],
                             help=DEGRADATIONS[kind]["help"])

    with st.spinner("Re-measuring the degraded audio…"):
        after = analyse(wav, kind, level, model_name)

    # --- before / after scores ------------------------------------------------
    b, a = base["score"], after["score"]
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"<div class='rp-card'><div class='rp-stat-label'>Original</div>"
                    f"<div class='rp-stat'>{b[1]:.2f} – {b[2]:.2f}</div>"
                    f"<div class='rp-muted'>central {b[0]:.2f}</div></div>",
                    unsafe_allow_html=True)
    with cols[1]:
        if a is None:
            st.markdown("<div class='rp-card'><div class='rp-stat-label'>Degraded</div>"
                        "<div class='rp-stat'>no score</div><div class='rp-muted'>"
                        "Praat could no longer find a pitch.</div></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='rp-card'><div class='rp-stat-label'>Degraded ({level})</div>"
                        f"<div class='rp-stat' style='color:{ACCENT}'>{a[1]:.2f} – {a[2]:.2f}</div>"
                        f"<div class='rp-muted'>central {a[0]:.2f}</div></div>",
                        unsafe_allow_html=True)
    with cols[2]:
        delta = "n/a" if a is None else f"{a[0] - b[0]:+.2f}"
        st.markdown(f"<div class='rp-card'><div class='rp-stat-label'>Change in central score</div>"
                    f"<div class='rp-stat'>{delta}</div><div class='rp-muted'>"
                    f"caused only by recording quality</div></div>", unsafe_allow_html=True)
    st.write("")

    # Before | after, side by side: listen, waveform, spectrogram.
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Original**")
        st.audio(base["wav"], format="audio/wav")
        show(fig_wave(base["traces"], TEXT_2, "Waveform"), key="chart_wave_before")
        show(fig_spec(base["traces"], "Spectrogram"), key="chart_spec_before")
    with right:
        st.markdown(f"**Degraded: {kind.lower()}, {level}**")
        st.audio(after["wav"], format="audio/wav")
        show(fig_wave(after["traces"], ACCENT, "Waveform"), key="chart_wave_after")
        show(fig_spec(after["traces"], "Spectrogram"), key="chart_spec_after")

    # --- which measurements moved -------------------------------------------
    rows = []
    for c in VOICE_FEATURES:
        bv, av = base["feats"][c], after["feats"][c]
        rows.append({"measurement": FEATURE_LABELS[c][0], "original": bv, "degraded": av,
                     "shift": (av - bv) / load_training_features()[c].std()})
    st.markdown("**Which measurements moved**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "original": st.column_config.NumberColumn(format="%.4g"),
        "degraded": st.column_config.NumberColumn(format="%.4g"),
        "shift": st.column_config.NumberColumn(
            "shift (× dataset spread)", format="%+.2f",
            help="Change divided by the standard deviation across the 81 dataset speakers. "
                 "±1 = as big as the typical difference between two people."),
    })

    with st.spinner("Scoring every severity level…"):
        curve = [analyse(wav, kind, lv, model_name)["score"] for lv in levels]
    show(fig_curve(levels, curve, levels.index(level)), key="chart_curve")
    st.caption("In our robustness test on held-out speakers, background noise pushed almost "
               "everyone (healthy or not) towards the Parkinson's side, and heavy compression "
               "pushed almost everyone towards the healthy side.")


def how_it_works(model_name: str):
    cv = load_models()["cv_results"]
    r = cv[(cv.subset == "All participants") & (cv.model == model_name)].set_index("features")
    voice = r.loc["Voice (18 features)"]
    st.markdown("---")
    st.markdown("### How this works and what it can't tell you")
    stats = st.columns(4)
    for col, (num, label) in zip(stats, [
        ("81", "telephone recordings (41 healthy, 40 Parkinson's)"),
        (f"{voice.acc_mean:.0%}", f"voice-model accuracy on unseen speakers (± {voice.acc_sd:.0%} across folds)"),
        ("51%", "accuracy from guessing"),
        (f"p ≈ {PERM_P_VOICE}", "voice model vs chance: not significant"),
    ]):
        col.markdown(f"<div class='rp-card'><div class='rp-stat'>{num}</div>"
                     f"<div class='rp-stat-label'>{label}</div></div>", unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            "**How it works.** Your audio is converted to WAV with ffmpeg, reduced to 8 kHz "
            "telephone quality to match the training data, and measured with Praat: pitch, "
            "pitch steadiness, jitter, shimmer, harmonics-to-noise ratio and 13 MFCCs. A "
            "logistic regression (or random forest) trained on the 81 recordings turns those "
            "18 numbers into a 0-1 score. Accuracy was measured with speaker-grouped "
            "cross-validation: no one was ever tested on a model that had heard them.")
    with right:
        st.markdown(
            f"**What we found, and what it can't tell you.**\n"
            f"- A model using only *recording* properties (length, loudness, noise floor) "
            f"scored {r.loc['Recording only (sanity check)'].acc_mean:.0%}, better than the voice "
            f"(p = {PERM_P_RECORDING}).\n"
            f"- **Age alone** scored {r.loc['Age only'].acc_mean:.0%}: the Parkinson's group "
            f"is about 20 years older.\n"
            f"- Noise and compression move the score more than the voice does.\n"
            f"- One vowel, one small dataset, phones of unknown type, no clinical validation.")
    st.caption(f"{PROTOTYPE}. Not a medical device. Full methods and citations are in README.md.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
TAB_NAMES = ["🎙️  Record / Upload", "📊  Your Voice vs. the Dataset",
             "🔊  How Noise Changes the Result"]


def delete_my_audio():
    """Button callback: forget this visitor's audio everywhere the app keeps it.

    1. Remove this input's entries from the in-memory caches: the converted WAV
       (prepare_audio) and every analysis of it (analyse) for every degradation
       level and both models. Other visitors' entries are untouched.
    2. Give the recorder and uploader new keys, so Streamlit discards the old
       widgets and the browser's copy of the recording.
    3. Reset the example toggle and degradation sliders, and show a confirmation.
    """
    active = st.session_state.pop("_active_input", None)
    if active is not None:
        raw, name, wav = active
        prepare_audio.clear(raw, name)
        if wav is not None:
            for model in ("logreg", "rf"):
                for kind, cfg in DEGRADATIONS.items():
                    for level in cfg["levels"]:
                        analyse.clear(wav, kind, level, model)
    for k in [k for k in st.session_state if k.startswith("deg_")]:
        del st.session_state[k]
    st.session_state.use_example = False          # also overrides ?example=1
    st.query_params.clear()
    st.session_state.input_version = st.session_state.get("input_version", 0) + 1
    st.session_state._deleted_flash = True


def main():
    st.set_page_config(page_title="Voice research prototype", page_icon="🔬", layout="wide")
    header()

    _, reset_col, right = st.columns([3.6, 1.4, 1])
    reset_slot = reset_col.empty()          # filled below once we know audio is loaded
    with right.popover("⚙︎ Model", width="stretch"):
        model_name = st.radio("Classifier", ["logreg", "rf"], key="model",
                              format_func={"logreg": "Logistic regression (default)",
                                           "rf": "Random forest"}.get)

    tab_names = TAB_NAMES
    # ?tab=2 or ?tab=3 in the URL opens that tab first.
    first = st.query_params.get("tab", "1")
    default = tab_names[int(first) - 1] if first in ("1", "2", "3") else tab_names[0]
    # Shown above the tabs so it's visible whichever tab the visitor was on.
    if st.session_state.pop("_deleted_flash", False):
        st.success("Your audio and its analysis were deleted from this app's memory. "
                   "Record or upload a new one in **Record / Upload**.", icon="🗑️")
    t1, t2, t3 = st.tabs(tab_names, key="main_tabs", default=default)

    # ---- Tab 1: input, checks, signal plots ---------------------------------
    wav, base, problem = None, None, None
    with t1:
        source = tab_record()
        if source is not None:
            raw, name = source
            # Remember exactly what was cached so "Delete my audio" can remove it.
            st.session_state._active_input = (raw, name, None)
            try:
                with st.spinner("Converting and measuring…"):
                    wav, notes = prepare_audio(raw, name)
                    st.session_state._active_input = (raw, name, wav)
                    base = analyse(wav, "Background noise", "None", model_name)
                    warnings = check_voicing(base["feats"], base["level_drop_db"])
            except AudioProblem as p:
                problem, wav, base = p, None, None
                show_problem(p)
            except Exception as e:                       # last-resort safety net
                problem, wav, base = AudioProblem("Something went wrong analysing this audio.",
                                                  f"Please try another recording. ({type(e).__name__})"), None, None
                show_problem(problem)

        if base is not None:
            f = base["feats"]
            chips = [("ok", f"✓ {f['voiced_duration_s']:.1f} s of voice"),
                     ("ok", f"✓ pitch ≈ {f['f0_mean_hz']:.0f} Hz"),
                     ("ok", f"✓ analysed at {TARGET_SR // 1000} kHz")]
            chips += [("warn", "! " + w.split(".")[0]) for w in warnings + notes]
            st.markdown("<div class='rp-chips'>" + "".join(
                f"<span class='rp-chip {k}'>{t}</span>" for k, t in chips) + "</div>",
                unsafe_allow_html=True)
            for w in warnings + notes:
                st.warning(w, icon="⚠️")
            st.caption("What the model hears (mono, 8 kHz telephone quality, volume-normalised):")
            st.audio(base["wav"], format="audio/wav")
            show(fig_signal(base["traces"]), key="chart_signal")
            st.info("Next: open **Your Voice vs. the Dataset** to see the score range and how "
                    "each measurement compares.", icon="👉")

    if source is not None:
        reset_slot.button("🗑️ Delete my audio", on_click=delete_my_audio,
                          width="stretch", key="delete_audio",
                          help="Done? Removes your recording and its analysis from this "
                               "app's memory and resets the page so you can start over.")

    # ---- Tabs 2 and 3 -------------------------------------------------------
    with t2:
        if base is None:
            # (plain if/else: a bare one-line conditional would be echoed to the
            #  page as "None" by Streamlit's magic)
            if problem:
                show_problem(problem)
            else:
                empty_state("how it compares with the dataset")
        else:
            tab_dataset(base, model_name)
    with t3:
        if base is None or base["score"] is None:
            if problem:
                show_problem(problem)
            else:
                empty_state("how recording quality moves the score")
        else:
            tab_noise(wav, base, model_name)

    how_it_works(model_name)


if __name__ == "__main__":
    main()
