from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


# ---- Paths ----
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent if THIS_DIR.name == "scripts" else THIS_DIR
AUDIO_DIR = PROJECT_ROOT / "audio"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---- Styling ----
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "grid.alpha": 0.3,
    }
)


def draw_pipeline_diagram(outfile: Path) -> None:
    """Draw the analysis pipeline flowchart."""
    labels = [
        "Generate canonical signals",
        "Load instrument recordings",
        "Select steady window",
        "Estimate f₀ and samples per period",
        "Extract cycles and isolate one period",
        "Compute Fourier coefficients (FFT)",
        "Truncate to |k| ≤ n and reconstruct Sₙ",
        "Compute MSE(n) and log–log fit for p",
    ]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    box_w, box_h = 2.6, 0.55
    gap = 0.2
    x0 = 0.1
    y0 = 0.5
    facecolor = "#e6e6e6"
    edgecolor = "#555555"
    textcolor = "#222222"

    boxes = []
    for i, text in enumerate(labels):
        x = x0 + i * (box_w + gap)
        rect = FancyBboxPatch(
            (x, y0 - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y0, text, ha="center", va="center", fontsize=9, color=textcolor, wrap=True)
        boxes.append((x, y0))

    for i in range(len(boxes) - 1):
        x_start = boxes[i][0] + box_w
        x_end = boxes[i + 1][0]
        ax.annotate(
            "",
            xy=(x_end - 0.02, y0),
            xytext=(x_start + 0.02, y0),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )

    ax.set_xlim(0, x0 + len(labels) * (box_w + gap))
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def select_steady_window(signal: np.ndarray, sr: int, window_duration: float = 0.35):
    """Centered steady window with bounds clamped to the signal."""
    n_sig = len(signal)
    n_win = int(window_duration * sr)
    if n_win > n_sig:
        n_win = n_sig
    mid = n_sig // 2
    start = max(0, mid - n_win // 2)
    end = min(n_sig, start + n_win)
    return signal[start:end], start, end


def plot_clarinet_steady_window() -> None:
    """Plot the clarinet A4 steady window."""
    audio_file = AUDIO_DIR / "BbClarinet.ff.A4.stereo.aif"
    if not audio_file.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_file}")

    signal, sr = librosa.load(audio_file, sr=None, mono=True)
    peak = np.max(np.abs(signal)) or 1.0
    signal = signal.astype(np.float32) / peak

    window, start_idx, end_idx = select_steady_window(signal, sr, 0.35)

    t = np.arange(len(signal)) / sr
    plt.figure(figsize=(9, 4))
    plt.plot(t, signal, color="#666666", linewidth=1.2, label="waveform")
    plt.axvspan(start_idx / sr, end_idx / sr, color="#b0c4de", alpha=0.5, label="steady window")
    plt.axvline(start_idx / sr, color="#1f78b4", linestyle="--", linewidth=1.0)
    plt.axvline(end_idx / sr, color="#1f78b4", linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised amplitude")
    plt.title("Clarinet A4 steady window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Fig3_2_ClarinetA4_SteadyWindow.png", dpi=300)
    plt.close()


def main() -> None:
    draw_pipeline_diagram(FIG_DIR / "Fig3_1_AnalysisPipeline.png")
    plot_clarinet_steady_window()


if __name__ == "__main__":
    main()
