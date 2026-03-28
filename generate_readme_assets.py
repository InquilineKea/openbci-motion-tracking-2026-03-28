#!/usr/bin/env python3
"""Generate static README visuals from a saved live OpenBCI payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}


def channel_sort_key(label: str) -> tuple[int, str]:
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits:
        return (0, f"{int(digits):03d}")
    return (1, label)


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_dashboard(payload: dict, out_path: Path) -> None:
    metrics = payload["metrics"]
    stream = payload["stream"]
    spectrum = payload["spectrum"]

    times = np.asarray(stream["time_axis_seconds"], dtype=float)
    freqs = np.asarray(spectrum["frequencies_hz"], dtype=float)
    median_psd = np.asarray(spectrum["median_psd"], dtype=float)
    eeg = {
        str(ch): np.asarray(values, dtype=float)
        for ch, values in (stream["eeg_uv"] or {}).items()
    }

    fig = plt.figure(figsize=(14, 8), dpi=150, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], height_ratios=[1.35, 1.0])
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_psd = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_bands = fig.add_subplot(gs[1, 1])

    offsets = np.linspace(0, 120, num=max(1, len(eeg)))
    for offset, (channel, values) in zip(offsets, sorted(eeg.items(), key=lambda item: channel_sort_key(item[0]))):
        if len(values) != len(times):
            continue
        ax_trace.plot(times, values + offset, linewidth=0.8, label=f"Ch {channel}")
    ax_trace.set_title("Rolling EEG Window", fontsize=12)
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("uV + offset")
    ax_trace.grid(alpha=0.2)
    ax_trace.legend(loc="upper right", fontsize=8, ncol=2)

    ax_psd.plot(freqs, median_psd, color="#0b7285", linewidth=2)
    for name, (lo, hi) in BANDS.items():
        ax_psd.axvspan(lo, hi, alpha=0.08, label=name if name in {"delta", "alpha", "gamma"} else None)
    ax_psd.set_xlim(0, min(45, float(freqs.max()) if freqs.size else 45))
    ax_psd.set_title("Median PSD", fontsize=12)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD")
    ax_psd.grid(alpha=0.2)

    ax_metrics.axis("off")
    artifact = "yes" if metrics.get("artifact_flag") else "no"
    lines = [
        "Live Metrics",
        "",
        f"Updated: {payload.get('updated_at', 'n/a')}",
        f"PAF: {metrics.get('paf_hz', float('nan')):.2f} Hz",
        f"Alpha/Theta: {metrics.get('alpha_theta_ratio', float('nan')):.3f}",
        f"Gamma/Delta: {metrics.get('gamma_delta_ratio', float('nan')):.3f}",
        f"1/f slope: {metrics.get('one_over_f_slope', float('nan')):.2f}",
        f"1/f r²: {metrics.get('one_over_f_r2', float('nan')):.2f}",
        f"Quality: {metrics.get('quality', 'unknown')}",
        f"Artifact: {artifact}",
        "",
        f"Channels: {', '.join(map(str, metrics.get('active_channels', [])))}",
        f"Window: {metrics.get('window_seconds', 'n/a')} s",
        f"Fs: {metrics.get('fs_hz', 'n/a')} Hz",
    ]
    ax_metrics.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=11,
        bbox={"facecolor": "#f8f9fa", "edgecolor": "#ced4da", "boxstyle": "round,pad=0.6"},
    )

    band_names = list(BANDS.keys())
    band_values = [band_power(freqs, median_psd, *BANDS[name]) for name in band_names]
    ax_bands.bar(band_names, band_values, color=["#4dabf7", "#74c0fc", "#ffd43b", "#ffa94d", "#ff6b6b"])
    ax_bands.set_title("Band Power Snapshot", fontsize=12)
    ax_bands.set_ylabel("Integrated PSD")
    ax_bands.grid(axis="y", alpha=0.2)

    fig.suptitle("OpenBCI Real-Time Dashboard Sample", fontsize=16, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(payload: dict, out_path: Path) -> None:
    metrics = payload["metrics"]
    spectrum = payload["spectrum"]
    freqs = np.asarray(spectrum["frequencies_hz"], dtype=float)
    median_psd = np.asarray(spectrum["median_psd"], dtype=float)
    by_channel = {
        str(ch): np.asarray(values, dtype=float)
        for ch, values in (spectrum["psd_by_channel"] or {}).items()
    }

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150, constrained_layout=True)
    for channel, psd in sorted(by_channel.items(), key=lambda item: channel_sort_key(item[0])):
        if len(psd) != len(freqs):
            continue
        ax.plot(freqs, psd, linewidth=0.9, alpha=0.35, label=f"Ch {channel}")
    ax.plot(freqs, median_psd, color="#111827", linewidth=2.5, label="Median PSD")

    colors = {
        "delta": "#4dabf7",
        "theta": "#74c0fc",
        "alpha": "#ffd43b",
        "beta": "#ffa94d",
        "gamma": "#ff6b6b",
    }
    ymax = float(np.nanmax(median_psd)) if median_psd.size else 1.0
    for name, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, color=colors[name], alpha=0.12)
        ax.text((lo + hi) / 2.0, ymax * 0.98, name, ha="center", va="top", fontsize=9, color=colors[name])

    paf = metrics.get("paf_hz")
    if isinstance(paf, (int, float)):
        ax.axvline(paf, color="#2b8a3e", linestyle="--", linewidth=1.5, label=f"PAF {paf:.2f} Hz")

    ax.set_xlim(0, min(45, float(freqs.max()) if freqs.size else 45))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Spectrum Snapshot", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    footer = (
        f"alpha/theta={metrics.get('alpha_theta_ratio', float('nan')):.3f}    "
        f"gamma/delta={metrics.get('gamma_delta_ratio', float('nan')):.3f}    "
        f"1/f slope={metrics.get('one_over_f_slope', float('nan')):.2f}    "
        f"quality={metrics.get('quality', 'unknown')}"
    )
    fig.text(0.5, 0.01, footer, ha="center", fontsize=10, family="monospace")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    payload = load_payload(args.payload_path)
    plot_dashboard(payload, args.out_dir / "eeg-dashboard-sample.png")
    plot_spectrum(payload, args.out_dir / "spectrum-sample.png")


if __name__ == "__main__":
    main()
