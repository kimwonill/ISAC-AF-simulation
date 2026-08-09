"""Regenerate the AF example with IEEE-style LaTeX text and math fonts.

The script reads the deterministic Fig. 2 realization produced by
``plot_af_simulation_example.m``, evaluates the corrected circular-correlation
ESL, and writes the submission PDF to both simulation and manuscript folders.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
PAPER_ROOT = ROOT.parent / "MyPaper"
DATA_PATH = ROOT / "results" / "fig2_af_simulation_example.mat"
OUTPUT_NAME = "af_no_coupling_combined.pdf"
TEX_BIN = Path(os.environ.get("APPDATA", "")) / "TinyTeX" / "bin" / "windows"


def setup_latex() -> None:
    """Use Times text and Computer-Modern math, matching IEEEtran output."""
    if not (TEX_BIN / "latex.exe").is_file():
        raise RuntimeError(f"LaTeX executable not found: {TEX_BIN / 'latex.exe'}")
    os.environ["PATH"] = f"{TEX_BIN}{os.pathsep}{os.environ.get('PATH', '')}"
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "text.latex.preamble": r"\renewcommand{\rmdefault}{ptm}",
        }
    )


def corrected_esl(power: np.ndarray, kappa: float) -> np.ndarray:
    """Closed-form ESL with the circular directional-power correlation."""
    power = np.asarray(power, dtype=float).reshape(-1)
    n_subcarriers = power.size
    squared_power_sum = np.sum(power**2)
    indices = np.arange(n_subcarriers)
    esl = np.empty((n_subcarriers, n_subcarriers), dtype=float)

    for delay in range(n_subcarriers):
        dft_value = np.sum(power * np.exp(-1j * 2 * np.pi * indices * delay / n_subcarriers))
        esl[delay, 0] = (kappa - 1) * squared_power_sum + abs(dft_value) ** 2
    for doppler in range(1, n_subcarriers):
        esl[:, doppler] = np.sum(power * np.roll(power, doppler))
    return np.maximum(esl, np.finfo(float).tiny)


def main() -> None:
    setup_latex()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run plot_af_simulation_example.m once to create the realization."
        )

    data = loadmat(DATA_PATH, squeeze_me=True, struct_as_record=False)
    power = data["P"]
    kappa = float(data["params"].kappa)
    esl = corrected_esl(power, kappa)
    esl_db = 10 * np.log10(esl / np.max(esl))
    color_min = -20.0

    n_subcarriers = esl.shape[0]
    delay = np.arange(n_subcarriers)
    doppler = np.arange(n_subcarriers)
    selected_doppler = [value for value in (1, 2, 4, 6, 8, 12, n_subcarriers - 1) if value < n_subcarriers]
    colors = ["#2f6fba", "#218a73", "#6d42ae", "#1a1a1a", "#ed8b00", "#007c91", "#666666"]

    sidelobe_map = np.ma.array(esl_db.T, mask=np.zeros_like(esl_db.T, dtype=bool))
    sidelobe_map.mask[0, 0] = True
    sidelobe_values = esl_db.copy()
    sidelobe_values[0, 0] = np.nan
    color_max = 5.0 * np.ceil(np.nanmax(sidelobe_values) / 5.0)
    cmap = mpl.colormaps["turbo"].copy()
    cmap.set_bad("black")

    nonzero_values = esl_db[:, 1:]
    cut_min = 0.5 * np.floor(np.min(nonzero_values) / 0.5)
    cut_max = 0.5 * np.ceil(np.max(nonzero_values) / 0.5)

    fig = plt.figure(figsize=(6.85, 3.25), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 0.055, 1), wspace=0.13)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_cut = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[0, 1])

    image = ax_map.imshow(
        sidelobe_map,
        origin="lower",
        aspect="equal",
        extent=(-0.5, n_subcarriers - 0.5, -0.5, n_subcarriers - 0.5),
        cmap=cmap,
        vmin=color_min,
        vmax=color_max,
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, cax=cax, ticks=np.linspace(color_min, color_max, 6))
    colorbar.ax.set_title(r"AF (dB)", pad=5, fontsize=9)
    ax_map.set_xlabel(r"Delay index $\tau$")
    ax_map.set_ylabel(r"Doppler index $\nu$")
    ax_map.set_xticks((0, 5, 10, 15))
    ax_map.set_yticks((0, 5, 10, 15))

    for value, color in zip(selected_doppler, colors):
        ax_cut.plot(delay, esl_db[:, value], color=color, linewidth=1.0, label=rf"$\nu={value}$")
    ax_cut.set_xlabel(r"Delay index $\tau$")
    ax_cut.set_ylabel(r"AF (dB)")
    ax_cut.set_xlim(-0.3, n_subcarriers - 0.7)
    ax_cut.set_ylim(cut_min, cut_max)
    ax_cut.set_yticks(np.arange(cut_min, cut_max + 0.01, 0.5))
    ax_cut.set_xticks((0, 5, 10, 15))
    ax_cut.grid(True, color="0.75", linewidth=0.45)
    ax_cut.legend(loc="lower center", ncol=2, frameon=True, framealpha=1.0,
                  edgecolor="0.2", handlelength=2.3, columnspacing=1.25)

    for axis, tag in ((ax_map, "(a)"), (ax_cut, "(b)")):
        axis.text(0.5, -0.24, tag, transform=axis.transAxes,
                  ha="center", va="top", fontsize=9.5)
        axis.tick_params(direction="in", top=True, right=True, length=3)

    simulation_output = ROOT / "figures" / OUTPUT_NAME
    manuscript_output = PAPER_ROOT / "figures" / OUTPUT_NAME
    simulation_output.parent.mkdir(exist_ok=True)
    manuscript_output.parent.mkdir(exist_ok=True)
    fig.savefig(simulation_output, format="pdf", bbox_inches="tight", pad_inches=0.015)
    shutil.copy2(simulation_output, manuscript_output)
    print(f"Wrote {simulation_output}")
    print(f"Copied {manuscript_output}")


if __name__ == "__main__":
    main()
