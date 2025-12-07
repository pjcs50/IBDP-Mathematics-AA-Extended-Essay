from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------- Configuration ---------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "grid.alpha": 0.3,
    }
)

COLOR_SQUARE = "#0a3b76"  # navy
COLOR_SAW = "#0f7c52"  # dark green
COLOR_TRI = "#8b1a1a"  # dark red
COLOR_GREY = "#555555"

THIS_DIR = Path(__file__).resolve().parent
# If the script lives in scripts/, use its parent as project root; otherwise stay here.
PROJECT_ROOT = THIS_DIR if (THIS_DIR / "audio").exists() else THIS_DIR.parent
AUDIO_DIR = PROJECT_ROOT / "audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR = OUTPUT_DIR / "figures"
for d in (AUDIO_DIR, CSV_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_CANON = 4096
N_MAX_CANON = min(400, N_CANON // 2 - 1)
CANON_N_MIN_FIT = 20
CANON_N_MAX_FIT = 200

N_MAX_INST_GLOBAL = 300
INST_N_MIN_FIT = 20
INST_N_MAX_FIT = 150

WINDOW_DURATION = 0.35
FLUTE_WINDOW_DURATION = 0.4  # default flute window
FLUTE_WINDOW_DURATION_A4 = 0.45  # extended window for flute A4
F0_FMIN = 50.0
F0_FMAX = 1500.0
CYCLES_DEFAULT = 4
FLUTE_CYCLES_CHOICES = (4, 3)  # prefer 4, else 3
FLUTE_N_MIN_FIT = 20
FLUTE_N_MAX_FIT = 40
FLUTE_MIN_FIT_POINTS = 15
FLUTE_FIT_BOUNDS = {
    "A4": (10, 24),
    "B3": (20, 40),
    "C4": (15, 30),
}
CLARINET_N_MIN_FIT = 30
CLARINET_N_MAX_FIT = 120
CLARINET_FIT_BOUNDS = {
    "A3": (30, 60),
    "A4": (30, 60),
    "C4": (30, 60),
}
NOISE_FLOOR_ABS = 1e-12
NOISE_FLOOR_REL = 1e-8

THEORETICAL_P = {
    "square": 1.0,  # |c_k| ~ 1/k, tail energy ~ 1/n
    "sawtooth": 1.0,
    "triangle": 3.0,  # |c_k| ~ 1/k^2, tail energy ~ 1/n^3
}


# --------- Helper functions ---------
def load_instrument_signal(path: str, instrument: str, note: str) -> Dict:
    """Load an audio file as mono float32 and normalise to max |x| = 1."""
    path = Path(path)
    signal, sr = librosa.load(path, sr=None, mono=True)
    peak = np.max(np.abs(signal)) or 1.0
    signal = signal.astype(np.float32) / peak
    return {
        "signal": signal,
        "sr": sr,
        "instrument": instrument,
        "note": note,
        "filename": path.name,
    }


def select_steady_window(
    signal: np.ndarray, sr: int, window_duration: float = WINDOW_DURATION
) -> Tuple[np.ndarray, int, int]:
    """
    Select a steady-state window centred in the signal.

    Returns the window and the start/end sample indices used.
    """
    n_sig = len(signal)
    n_win = int(window_duration * sr)
    if n_win > n_sig:
        n_win = n_sig
    mid = n_sig // 2
    start = max(0, mid - n_win // 2)
    end = min(n_sig, start + n_win)
    window = signal[start:end]
    return window, start, end


def estimate_fundamental_frequency(
    segment: np.ndarray, sr: int, fmin: float = F0_FMIN, fmax: float = F0_FMAX
) -> float:
    """Estimate f0 via FFT peak search within [fmin, fmax]."""
    if len(segment) == 0:
        return 0.0
    windowed = segment * np.hanning(len(segment))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sr)
    mag = np.abs(spectrum)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    idx = np.argmax(mag[mask])
    return float(freqs[mask][idx])


def extract_one_period(
    segment: np.ndarray, sr: int, f0: float, cycles: int = CYCLES_DEFAULT
) -> np.ndarray:
    """Extract a centred chunk containing several cycles."""
    if f0 <= 0:
        return segment
    n0 = max(1, int(round(sr / f0)))
    n = cycles * n0
    if n > len(segment):
        cycles = max(1, len(segment) // n0)
        n = max(n0, cycles * n0)
        n = min(n, len(segment))
    start = max(0, len(segment) // 2 - n // 2)
    end = min(len(segment), start + n)
    return segment[start:end]


def make_periodic(signal: np.ndarray, n0: int) -> np.ndarray:
    """Return one period (first n0 samples); truncates if needed."""
    n_use = min(n0, len(signal))
    return signal[:n_use]


def average_cycles(segment: np.ndarray, n0: int) -> np.ndarray:
    """Average full cycles within segment to reduce noise; returns one period."""
    cycles_available = len(segment) // n0
    if cycles_available < 1:
        return make_periodic(segment, n0)
    usable = cycles_available * n0
    trimmed = segment[:usable]
    reshaped = trimmed.reshape(cycles_available, n0)
    return reshaped.mean(axis=0)


def compute_fft_coeffs(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute discrete Fourier coefficients c_k and integer harmonic indices."""
    n = len(f)
    c_k = np.fft.fft(f) / n
    k_indices = (np.fft.fftfreq(n, d=1.0) * n).astype(int)
    return c_k, k_indices


def partial_sum_from_coeffs(c_k: np.ndarray, k_indices: np.ndarray, n: int) -> np.ndarray:
    """Build S_n by zeroing coefficients with |k| > n and inverse FFT."""
    mask = np.abs(k_indices) <= n
    truncated = c_k * mask
    return np.fft.ifft(truncated * len(c_k)).real


def mse_vs_n(f: np.ndarray, c_k: np.ndarray, k_indices: np.ndarray, n_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MSE(f, S_n) for n = 1..n_max."""
    n_vals = np.arange(1, n_max + 1)
    mse_vals = np.zeros_like(n_vals, dtype=float)
    for i, n in enumerate(n_vals):
        s_n = partial_sum_from_coeffs(c_k, k_indices, n)
        mse_vals[i] = float(np.mean((f - s_n) ** 2))
    return n_vals, mse_vals


def fit_decay_exponent(
    n_vals: np.ndarray, mse_vals: np.ndarray, n_min: int, n_max: int
) -> Dict:
    """Fit log–log slope of MSE vs n; returns p_hat = -slope."""
    mask = (n_vals >= n_min) & (n_vals <= n_max)
    if not np.any(mask):
        return {
            "p_hat": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n_min": int(n_min),
            "n_max": int(n_max),
        }
    x = np.log(n_vals[mask])
    y = np.log(mse_vals[mask])
    if x.size < 2:
        return {
            "p_hat": np.nan,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "n_min": int(n_min),
            "n_max": int(n_max),
        }
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "p_hat": float(-slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_min": int(n_min),
        "n_max": int(n_max),
    }


# --------- Plotting helpers ---------
def plot_window(signal: np.ndarray, sr: int, start: int, end: int, outfile: str, title: str) -> None:
    """Plot waveform with shaded steady window."""
    t = np.arange(len(signal)) / sr
    plt.figure(figsize=(9, 4))
    plt.plot(t, signal, color=COLOR_GREY, linewidth=1.2)
    plt.axvspan(start / sr, end / sr, color="#b0c4de", alpha=0.5, label="steady window")
    plt.axvline(start / sr, color="#1f78b4", linestyle="--", linewidth=1)
    plt.axvline(end / sr, color="#1f78b4", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised amplitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_spectrum(k_pos: np.ndarray, mag: np.ndarray, outfile: str, title: str) -> None:
    """Plot magnitude spectrum on log–log axes."""
    plt.figure(figsize=(8, 5))
    plt.loglog(k_pos, mag, marker="o", linestyle="-", color=COLOR_SQUARE, linewidth=1.2, markersize=3)
    plt.xlabel("Harmonic number k")
    plt.ylabel("|c_k|")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_mse_with_fit(
    n_vals: np.ndarray,
    mse_vals: np.ndarray,
    fit: Dict,
    outfile: str,
    title: str,
    color: str,
) -> None:
    """Plot MSE points and fitted power-law line."""
    plt.figure(figsize=(8, 5))
    plt.loglog(n_vals, mse_vals, marker="o", linestyle="", color=color, markersize=4, label="data")
    mask = (n_vals >= fit["n_min"]) & (n_vals <= fit["n_max"])
    n_fit = n_vals[mask]
    fit_line = np.exp(fit["intercept"]) * n_fit ** (fit["slope"])
    plt.loglog(n_fit, fit_line, linestyle="--", color=color, linewidth=1.2, label="fit")
    plt.xlabel("Number of harmonics n")
    plt.ylabel("MSE(n)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_triangle_coeff_decay(k_pos: np.ndarray, c_pos: np.ndarray, outfile: str) -> None:
    """Plot |c_k| decay for triangle with slope -2 reference line."""
    plt.figure(figsize=(8, 5))
    mag = np.abs(c_pos)
    plt.loglog(k_pos, mag, marker="o", linestyle="-", color=COLOR_TRI, markersize=3, linewidth=1.2, label="|c_k|")
    if len(k_pos) > 1:
        k_ref = k_pos[1]
        y_ref = mag[1]
        ref_line = y_ref * (k_pos / k_ref) ** (-2)
        plt.loglog(k_pos, ref_line, linestyle="--", color=COLOR_GREY, linewidth=1.0, label="slope -2 ref")
    plt.xlabel("Harmonic number k")
    plt.ylabel("|c_k| (log scale)")
    plt.title("Triangle wave coefficient decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_canonical_mse(
    curves: List[Tuple[str, np.ndarray, np.ndarray, Dict]], outfile: str
) -> None:
    """Plot MSE decay for canonical signals with fitted lines."""
    plt.figure(figsize=(8, 5))
    color_map = {"square": COLOR_SQUARE, "sawtooth": COLOR_SAW, "triangle": COLOR_TRI}
    for name, n_vals, mse_vals, fit in curves:
        col = color_map.get(name, COLOR_GREY)
        plt.loglog(n_vals, mse_vals, marker="o", linestyle="", markersize=3, color=col, label=name)
        mask = (n_vals >= fit["n_min"]) & (n_vals <= fit["n_max"])
        n_fit = n_vals[mask]
        fit_line = np.exp(fit["intercept"]) * n_fit ** (fit["slope"])
        plt.loglog(n_fit, fit_line, linestyle="--", linewidth=1.2, color=col)
    plt.xlabel("Number of harmonics n")
    plt.ylabel("MSE(n)")
    plt.title("Canonical MSE decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


# --------- Analysis pipelines ---------
def build_canonical_signals() -> Dict[str, np.ndarray]:
    """Construct square, sawtooth, and triangle waves on [0, 2π)."""
    t = np.linspace(0, 2 * np.pi, N_CANON, endpoint=False)
    square = np.where((t > 0) & (t < np.pi), 1.0, -1.0)
    sawtooth = (t / np.pi) - 1.0
    triangle = np.zeros_like(t)
    rising = (t >= 0) & (t < np.pi / 2)
    falling = (t >= np.pi / 2) & (t < 3 * np.pi / 2)
    rising2 = (t >= 3 * np.pi / 2) & (t < 2 * np.pi)
    triangle[rising] = (2 / np.pi) * t[rising]
    triangle[falling] = 2 - (2 / np.pi) * t[falling]
    triangle[rising2] = -4 + (2 / np.pi) * t[rising2]
    return {"square": square, "sawtooth": sawtooth, "triangle": triangle}


def canonical_analysis() -> None:
    signals = build_canonical_signals()
    summary_rows = []
    mse_curves = []
    triangle_coeff = None

    for name, f in signals.items():
        c_k, k_idx = compute_fft_coeffs(f)
        n_vals, mse_vals = mse_vs_n(f, c_k, k_idx, N_MAX_CANON)
        fit = fit_decay_exponent(n_vals, mse_vals, CANON_N_MIN_FIT, CANON_N_MAX_FIT)
        mse_curves.append((name, n_vals, mse_vals, fit))
        df = pd.DataFrame({"n": n_vals, "MSE": mse_vals})
        df.to_csv(CSV_DIR / f"canonical_{name}_MSE.csv", index=False)
        summary_rows.append(
            {
                "waveform": name,
                "n_min_used": fit["n_min"],
                "n_max_used": fit["n_max"],
                "p_hat": fit["p_hat"],
                "r2": fit["r2"],
            }
        )
        if name == "triangle":
            mask = k_idx > 0
            triangle_coeff = (k_idx[mask], c_k[mask])

        theo_p = THEORETICAL_P.get(name)
        if theo_p is not None:
            diff = abs(theo_p - fit["p_hat"])
            print(f"{name:9s} theoretical p={theo_p:.2f}, fitted p={fit['p_hat']:.3f}, |Δ|={diff:.3f}, r2={fit['r2']:.3f}")

    pd.DataFrame(summary_rows).to_csv(CSV_DIR / "canonical_summary_p.csv", index=False)

    if triangle_coeff:
        k_pos, c_pos = triangle_coeff
        plot_triangle_coeff_decay(k_pos, c_pos, FIG_DIR / "Fig3_1_TriangleCoeffDecay.png")

    plot_canonical_mse(mse_curves, FIG_DIR / "Fig3_3_Canonical_MSE_Decay.png")


def instrument_analysis(process_clarinet: bool = True, process_flute: bool = True) -> None:
    files = []
    if process_clarinet:
        files.extend(
            [
                ("BbClarinet.ff.A3.stereo.aif", "clarinet", "A3"),
                ("BbClarinet.ff.A4.stereo.aif", "clarinet", "A4"),
                ("BbClarinet.ff.C4.stereo.aif", "clarinet", "C4"),
            ]
        )
    if process_flute:
        files.extend(
            [
                ("Flute.nonvib.ff.A4.stereo.aif", "flute", "A4"),
                ("Flute.nonvib.ff.B3.stereo.aif", "flute", "B3"),
                ("Flute.nonvib.ff.C4.stereo.aif", "flute", "C4"),
            ]
        )

    summary_rows = []
    mse_store = {}

    for fname, instrument, note in files:
        candidates = [AUDIO_DIR / fname, PROJECT_ROOT / fname]
        file_path = next((p for p in candidates if p.exists()), None)
        if file_path is None:
            print(f"Skipping missing file: {fname}")
            continue

        data = load_instrument_signal(file_path, instrument, note)
        signal = data["signal"]
        sr = data["sr"]

        if instrument == "flute":
            window_duration = FLUTE_WINDOW_DURATION_A4 if note == "A4" else FLUTE_WINDOW_DURATION
        else:
            window_duration = WINDOW_DURATION

        window, w_start, w_end = select_steady_window(signal, sr, window_duration)
        f0 = estimate_fundamental_frequency(window, sr, F0_FMIN, F0_FMAX)
        if f0 <= 0:
            print(f"Warning: f0 not found for {fname}; defaulting to length-based f0.")
            f0 = sr / max(1, len(window))

        n0 = max(1, int(round(sr / f0)))
        cycles_to_use = CYCLES_DEFAULT
        if instrument == "flute":
            if len(window) >= FLUTE_CYCLES_CHOICES[0] * n0:
                cycles_to_use = FLUTE_CYCLES_CHOICES[0]
            elif len(window) >= FLUTE_CYCLES_CHOICES[1] * n0:
                cycles_to_use = FLUTE_CYCLES_CHOICES[1]
        segment = extract_one_period(window, sr, f0, cycles_to_use)
        period = make_periodic(segment, n0)
        n_period = len(period)
        n_max_inst = min(N_MAX_INST_GLOBAL, n_period // 2 - 1) if n_period > 2 else 1
        if n_max_inst < 1:
            print(f"Skipping {fname}: not enough samples for Fourier analysis.")
            continue

        c_k, k_idx = compute_fft_coeffs(period)
        n_vals, mse_vals = mse_vs_n(period, c_k, k_idx, n_max_inst)

        noise_floor = max(NOISE_FLOOR_ABS, NOISE_FLOOR_REL * float(np.max(mse_vals)))
        valid_mask = mse_vals > noise_floor
        n_vals = n_vals[valid_mask]
        mse_vals = mse_vals[valid_mask]

        if instrument == "flute":
            bounds = FLUTE_FIT_BOUNDS.get(note, (FLUTE_N_MIN_FIT, FLUTE_N_MAX_FIT))
            fit_lower = bounds[0]
            fit_upper = min(bounds[1], int(n_vals.max()) if n_vals.size else bounds[1])
            available = int(np.sum((n_vals >= fit_lower) & (n_vals <= fit_upper)))
            while available < FLUTE_MIN_FIT_POINTS and fit_lower > 10:
                fit_lower = max(10, fit_lower - 1)
                available = int(np.sum((n_vals >= fit_lower) & (n_vals <= fit_upper)))
            if available < FLUTE_MIN_FIT_POINTS or fit_lower >= fit_upper:
                fit = {
                    "p_hat": np.nan,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "n_min": fit_lower,
                    "n_max": fit_upper,
                }
            else:
                fit = fit_decay_exponent(n_vals, mse_vals, fit_lower, fit_upper)
        else:
            fit_bounds = CLARINET_FIT_BOUNDS.get(note, (CLARINET_N_MIN_FIT, CLARINET_N_MAX_FIT))
            fit_lower = fit_bounds[0]
            fit_upper = min(fit_bounds[1], n_max_inst)
            available = int(np.sum((n_vals >= fit_lower) & (n_vals <= fit_upper)))
            if available < 12 or fit_lower >= fit_upper:
                fit = {
                    "p_hat": np.nan,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                    "n_min": fit_lower,
                    "n_max": fit_upper,
                }
            else:
                fit = fit_decay_exponent(n_vals, mse_vals, fit_lower, fit_upper)

        instrument_title = instrument.capitalize()
        mse_outfile = CSV_DIR / f"{instrument_title}_{note}_MSE.csv"
        pd.DataFrame({"n": n_vals, "MSE": mse_vals}).to_csv(mse_outfile, index=False)

        summary_rows.append(
            {
                "instrument": instrument.capitalize(),
                "note": note,
                "p_hat": fit["p_hat"],
                "r2": fit["r2"],
                "n_min_fit": fit["n_min"],
                "n_max_fit": fit["n_max"],
            }
        )

        key = f"{instrument_title}_{note}"
        mse_store[key] = {"n_vals": n_vals, "mse_vals": mse_vals, "fit": fit}

        # Plots: keep original clarinet A4 & flute C4, and also per-flute-note MSE
        if key in {"Clarinet_A4", "Flute_C4"}:
            wave_out = FIG_DIR / (
                f"Fig3_2{'B' if instrument=='flute' else ''}_{instrument_title}_{note}_WaveformWindow.png"
            )
            plot_window(signal, sr, w_start, w_end, wave_out, f"{instrument_title} {note} steady window")
            mask = k_idx > 0
            k_pos = k_idx[mask]
            c_pos = c_k[mask]
            spec_out = FIG_DIR / (
                "Fig4_1_Clarinet_A4_Spectrum.png" if instrument == "clarinet" else "Fig4_2_Flute_C4_Spectrum.png"
            )
            plot_spectrum(k_pos, np.abs(c_pos), spec_out, f"{instrument_title} {note} spectrum")
            pd.DataFrame({"k": k_pos, "abs_c_k": np.abs(c_pos)}).to_csv(
                CSV_DIR / f"{instrument_title}_{note}_Spectrum.csv", index=False
            )

        if instrument == "flute":
            plot_mse_with_fit(
                n_vals,
                mse_vals,
                fit,
                FIG_DIR / f"Fig4_4_Flute_{note}_MSE.png",
                f"{instrument_title} {note} MSE decay",
                COLOR_SAW,
            )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            CSV_DIR / "instrument_summary_p.csv", index=False, columns=["instrument", "note", "p_hat", "r2", "n_min_fit", "n_max_fit"]
        )

    # Combined clarinet vs flute MSE plot if available
    if "Clarinet_A4" in mse_store and "Flute_C4" in mse_store:
        plt.figure(figsize=(8, 5))
        for key, color in [("Clarinet_A4", COLOR_SQUARE), ("Flute_C4", COLOR_SAW)]:
            curve = mse_store[key]
            n_vals = curve["n_vals"]
            mse_vals = curve["mse_vals"]
            fit = curve["fit"]
            plt.loglog(n_vals, mse_vals, marker="o", linestyle="", markersize=3, color=color, label=key.replace("_", " "))
            mask = (n_vals >= fit["n_min"]) & (n_vals <= fit["n_max"])
            n_fit = n_vals[mask]
            fit_line = np.exp(fit["intercept"]) * n_fit ** (fit["slope"])
            plt.loglog(n_fit, fit_line, linestyle="--", linewidth=1.2, color=color)
        plt.xlabel("Number of harmonics n")
        plt.ylabel("MSE(n)")
        plt.title("MSE decay: Clarinet A4 vs Flute C4")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "Fig4_5_MSE_Clarinet_vs_Flute.png", dpi=300)
        plt.close()


def main() -> None:
    canonical_analysis()
    instrument_analysis(process_clarinet=True, process_flute=True)


if __name__ == "__main__":
    main()
