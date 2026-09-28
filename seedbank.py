# -*- coding: utf-8 -*-
"""Reproduce the precipitation and three-species seed-bank figure.

Extreme precipitation is classified from DESST January-March totals during
1965-2018; the seed-bank simulation retains the 30 years 1984-2013.
The figure places precipitation above a shared log-scale abundance panel.

The DESST precipitation record was obtained from the Santa Rita Experimental
Range. Source values are recorded in hundredths of an inch and converted to
millimetres here. Species parameters are based on the published sources cited
in the accompanying manuscript. The precipitation record is used as an
independent regional environmental forcing; the simulation is not a
reconstruction of population dynamics at the original demographic study site.

N_t denotes seed-bank density at the beginning of year t. Precipitation P_t
determines N_{t+1}; therefore, the plotted 1984-2013 seed-bank series uses the
1984-2012 precipitation values. The 2013 precipitation value would determine
N_2014, which is outside the plotted seed-bank period.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
PRECIP_FILE = PROJECT_DIR / "data" / "precip1922_2017.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Journal-style typography and line settings
# -----------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Quantile-based precipitation classification used in this study
# Reference period: DESST January-March precipitation, 1965-2018.
# Simulation period: 1984-2013 (30 years).
# -----------------------------------------------------------------------------
REFERENCE_START = 1965
REFERENCE_END = 2018
TARGET_START = 1984
TARGET_END = 2013
STATION = "DESST"
SEASON_MONTHS = ("JAN", "FEB", "MAR")
RAW_MISSING = -9999
RAW_TO_MM = 0.254  # source units are hundredths of an inch

historical_rainfall = {}
with PRECIP_FILE.open("r", newline="", encoding="utf-8-sig") as stream:
    reader = csv.DictReader(stream)
    required = {"STATION", "YEAR", *SEASON_MONTHS}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(f"Precipitation file lacks required columns: {sorted(required)}")
    for row in reader:
        if row["STATION"].strip() != STATION:
            continue
        year = int(row["YEAR"])
        monthly = [int(float(row[month])) for month in SEASON_MONTHS]
        if any(value == RAW_MISSING for value in monthly):
            continue
        if any(value < 0 for value in monthly):
            raise ValueError(f"Unexpected negative precipitation in {year}: {monthly}")
        historical_rainfall[year] = sum(monthly) * RAW_TO_MM

reference_years = [
    year for year in range(REFERENCE_START, REFERENCE_END + 1)
    if year in historical_rainfall
]
if len(reference_years) < 30:
    raise ValueError("Fewer than 30 complete years in the reference period.")

reference_values = np.asarray(
    [historical_rainfall[year] for year in reference_years], dtype=float
)
q10, q25, q75, q90 = np.quantile(
    reference_values, [0.10, 0.25, 0.75, 0.90]
)


def classify_year(precipitation):
    """Classify seasonal precipitation using historical quantile thresholds."""
    if precipitation < q10:
        return "extreme dry"
    if precipitation < q25:
        return "moderate dry"
    if precipitation <= q75:
        return "normal"
    if precipitation <= q90:
        return "moderate wet"
    return "extreme wet"


year_labels = np.arange(TARGET_START, TARGET_END + 1)
missing_target_years = [
    int(year) for year in year_labels if int(year) not in historical_rainfall
]
if missing_target_years:
    raise ValueError(f"Incomplete target years: {missing_target_years}")

real_rainfall = np.asarray(
    [historical_rainfall[int(year)] for year in year_labels], dtype=float
)
n_real = real_rainfall.size
if n_real != 30:
    raise ValueError(f"Expected 30 simulation years, found {n_real}.")
P_mean = real_rainfall.mean()

event_class = np.asarray(
    [classify_year(value) for value in real_rainfall], dtype=object
)
extreme_drought_mask = event_class == "extreme dry"
moderate_drought_mask = event_class == "moderate dry"
normal_mask = event_class == "normal"
moderate_wet_mask = event_class == "moderate wet"
extreme_wet_mask = event_class == "extreme wet"
drought_mask = extreme_drought_mask
wet_mask = extreme_wet_mask
disturbance_mask = drought_mask | wet_mask

isolated_mask = np.zeros(n_real, dtype=bool)
for t in range(1, n_real - 1):
    isolated_mask[t] = (
        normal_mask[t - 1]
        and disturbance_mask[t]
        and normal_mask[t + 1]
    )
isolated_indices = np.where(isolated_mask)[0]

print(
    f"Reference: {REFERENCE_START}-{REFERENCE_END}, "
    f"{len(reference_years)} complete DESST Jan-Mar years"
)
print(
    f"Thresholds (mm): q10={q10:.3f}, q25={q25:.3f}, "
    f"q75={q75:.3f}, q90={q90:.3f}"
)
print(f"Simulation period: {TARGET_START}-{TARGET_END} ({n_real} years)")
print("Strict normal-extreme-normal years:", year_labels[isolated_indices].tolist())


# -----------------------------------------------------------------------------
# Seed-bank model. H_i is already the effective seed yield used in the paper.
# -----------------------------------------------------------------------------
def equilibrium(H_i, g_i, A_i, d_i, precipitation=P_mean):
    carryover = (1.0 - g_i) * (1.0 - d_i)
    return (
        (precipitation * g_i * H_i) / (1.0 - carryover)
        - precipitation
    ) / (A_i * g_i)


def simulate(N0, H_i, g_i, A_i, d_i):
    """Simulate beginning-of-year seed-bank density from 1984 to 2013.

    Precipitation in year t updates N_t to N_{t+1}. The final precipitation
    value (2013) is displayed but is not used because N_2014 is not plotted.
    """
    abundance = np.zeros(real_rainfall.size, dtype=float)
    abundance[0] = N0
    for t in range(real_rainfall.size - 1):
        precipitation = real_rainfall[t]
        recruitment = (
            precipitation * g_i
            / (precipitation + A_i * g_i * abundance[t])
            * H_i
        )
        carryover = (1.0 - g_i) * (1.0 - d_i)
        abundance[t + 1] = abundance[t] * (recruitment + carryover)
    return abundance


# H_i values below are effective yields: H_raw multiplied by new-seed survival.
species_parameters = {
    "MOBE": {"g": 0.648, "d": 0.727, "H": 35.2 * 0.102, "A": 4.0},
    "STMI": {"g": 0.419, "d": 0.542, "H": 38.2 * 0.145, "A": 4.0},
    "EVMU": {"g": 0.088, "d": 0.172, "H": 112.7 * 0.214, "A": 3.0},
}

abundance_by_species = {}
for species_name, values in species_parameters.items():
    N0 = equilibrium(values["H"], values["g"], values["A"], values["d"])
    abundance_by_species[species_name] = simulate(
        N0, values["H"], values["g"], values["A"], values["d"]
    )


# -----------------------------------------------------------------------------
# Shared plotting helpers
# -----------------------------------------------------------------------------
species_styles = {
    "MOBE": dict(color="black", linestyle="-", marker="o"),
    "STMI": dict(color="black", linestyle="--", marker="s"),
    "EVMU": dict(color="0.35", linestyle="-.", marker="^"),
}


def style_axis(ax):
    ax.tick_params(direction="in", top=False, right=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def draw_precipitation(ax, panel_label):
    colors = np.full(real_rainfall.size, "0.78", dtype=object)
    colors[moderate_drought_mask | moderate_wet_mask] = "0.60"
    colors[extreme_drought_mask | extreme_wet_mask] = "0.35"
    colors[isolated_mask] = "0.12"
    ax.bar(
        year_labels,
        real_rainfall,
        width=0.78,
        color=colors.tolist(),
        edgecolor="none",
        zorder=2,
    )
    ax.axhline(q10, color="0.25", linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(q90, color="0.25", linestyle=":", linewidth=0.9, zorder=1)
    ax.text(
        2013.25, q10, f"q10={q10:.1f} mm", ha="right", va="bottom",
        fontsize=7.5, color="0.25"
    )
    ax.text(
        2013.25, q90, f"q90={q90:.1f} mm", ha="right", va="bottom",
        fontsize=7.5, color="0.25"
    )
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlim(1983.5, 2013.5)
    ax.set_ylim(0.0, 300.0)
    ax.set_yticks([0.0, 100.0, 200.0, 300.0])
    ax.set_title(
        f"{panel_label}  Precipitation",
        loc="left", pad=3, fontweight="bold"
    )
    ax.legend(
        handles=[
            Patch(facecolor="0.25", edgecolor="none", label="Strict normal-extreme-normal event")
        ],
        loc="upper right", frameon=False, handlelength=1.2,
        borderaxespad=0.4
    )
    style_axis(ax)


def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Design A: precipitation plus one shared log-scale abundance panel
# -----------------------------------------------------------------------------
fig_a, (ax_a_rain, ax_a_abundance) = plt.subplots(
    2,
    1,
    figsize=(7.2, 4.7),
    sharex=True,
    gridspec_kw={"height_ratios": [0.9, 2.2], "hspace": 0.08},
)
draw_precipitation(ax_a_rain, "(a)")
ax_a_rain.tick_params(labelbottom=False)

for species_name in ("MOBE", "STMI", "EVMU"):
    style = species_styles[species_name]
    ax_a_abundance.plot(
        year_labels,
        abundance_by_species[species_name],
        label=species_name,
        linewidth=1.35,
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.75,
        markevery=3,
        **style,
    )

all_abundance = np.concatenate(list(abundance_by_species.values()))
ax_a_abundance.set_yscale("log")
ax_a_abundance.set_ylim(all_abundance.min() * 0.82, all_abundance.max() * 1.22)
ax_a_abundance.yaxis.set_major_locator(FixedLocator([100.0, 1000.0, 10000.0]))
ax_a_abundance.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax_a_abundance.yaxis.set_minor_locator(NullLocator())
ax_a_abundance.tick_params(axis="y", which="minor", left=False, right=False)

fig_a.canvas.draw()

for tick in ax_a_abundance.yaxis.get_major_ticks():
    if np.isclose(tick.get_loc(), 1e2):
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)

ax_a_abundance.set_xlim(1983.5, 2013.5)
ax_a_abundance.set_xticks([1985, 1990, 1995, 2000, 2005, 2010])
ax_a_abundance.set_xlabel("Year")
ax_a_abundance.set_ylabel(r"Seed-bank density, $N_{i,t}$")
ax_a_abundance.set_title(
    "(b)  Soil seed-bank density",
    loc="left", pad=3, fontweight="bold"
)
ax_a_abundance.legend(
    loc="center right", bbox_to_anchor=(0.99, 0.63),
    frameon=False, handlelength=2.6,
    labelspacing=0.55, borderaxespad=0.7
)
style_axis(ax_a_abundance)

fig_a.align_ylabels()
fig_a.subplots_adjust(left=0.105, right=0.985, bottom=0.115, top=0.97)
save_figure(fig_a, "figure1_seedbank_dynamics")


print("Generated Figure 1:")
print(OUTPUT_DIR / "figure1_seedbank_dynamics.pdf")
print("Isolated precipitation-event years:", year_labels[isolated_indices].tolist())
print("Marked abundance-response years (t+1):", year_labels[isolated_indices + 1].tolist())
