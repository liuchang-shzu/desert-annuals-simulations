"""Reproduce the seed-bank sensitivity analysis reported in the manuscript.

DESST January-March precipitation is classified using quantiles calculated
from complete seasons in 1965-2018. The simulation period is 1984-2013.
Species-specific germination and seed-survival parameters are based on Cuello
et al. (2019) and Gremer et al. (2014), as described in the manuscript.

The DESST record is used as an independent regional environmental forcing;
the simulations are not reconstructions of population dynamics at the
original demographic study site.

N_t denotes seed-bank density at the beginning of year t. Precipitation P_t
determines N_{t+1}; therefore, the 1984-2013 seed-bank series uses the
1984-2012 precipitation values. The 2013 precipitation value would determine
N_2014, which is outside the plotted seed-bank period.
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter, NullFormatter
import csv
from pathlib import Path

# ============================================================
# Load seasonal precipitation and classify precipitation years
# ============================================================
print("Loading DESST January-March precipitation records...")

PROJECT_DIR = Path(__file__).resolve().parent
REFERENCE_STATION = 'DESST'
SEASON_MONTHS = ('JAN', 'FEB', 'MAR')
RAW_MISSING = -9999
RAW_TO_MM = 0.254  # Source values are hundredths of an inch.
REFERENCE_START = 1965
REFERENCE_END = 2018
TARGET_START = 1984
TARGET_END = 2013
year_labels = np.arange(TARGET_START, TARGET_END + 1)
REFERENCE_PRECIP_FILE = PROJECT_DIR / "data" / "precip1922_2017.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

historical_rainfall = {}
with REFERENCE_PRECIP_FILE.open(
    "r", newline="", encoding="utf-8-sig"
) as stream:
    reader = csv.DictReader(stream)
    required = {'STATION', 'YEAR', *SEASON_MONTHS}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError(
            f'Precipitation file lacks required columns: {sorted(required)}'
        )
    for row in reader:
        if row['STATION'].strip() != REFERENCE_STATION:
            continue
        year = int(row['YEAR'])
        monthly = [int(float(row[m])) for m in SEASON_MONTHS]
        if any(value == RAW_MISSING for value in monthly):
            continue
        if any(value < 0 for value in monthly):
            raise ValueError(
                f'Unexpected negative precipitation in {year}: {monthly}'
            )
        historical_rainfall[year] = sum(monthly) * RAW_TO_MM

reference_years = [
    y for y in range(REFERENCE_START, REFERENCE_END + 1)
    if y in historical_rainfall
]
if len(reference_years) < 30:
    raise ValueError(
        'Fewer than 30 complete years occur in the reference period.'
    )

missing_target_years = [y for y in year_labels if y not in historical_rainfall]
if missing_target_years:
    raise ValueError(f'Incomplete simulation years: {missing_target_years}')

reference_values = np.asarray(
    [historical_rainfall[y] for y in reference_years], dtype=float
)
q10, q25, q75, q90 = np.quantile(
    reference_values, [0.10, 0.25, 0.75, 0.90]
)
def classify_year(precipitation):
    """Classify seasonal precipitation using historical quantile thresholds."""
    if precipitation < q10:
        return 'Extreme drought'
    if precipitation < q25:
        return 'Moderate drought'
    if precipitation <= q75:
        return 'Normal'
    if precipitation <= q90:
        return 'Moderate wet'
    return 'Extreme wet'


real_rainfall = np.asarray(
    [historical_rainfall[y] for y in year_labels], dtype=float
)
n_real = len(real_rainfall)
P_mean = real_rainfall.mean()

event_class = np.asarray([classify_year(P) for P in real_rainfall], dtype=object)
extreme_drought_mask = event_class == 'Extreme drought'
moderate_drought_mask = event_class == 'Moderate drought'
climate_normal_mask = event_class == 'Normal'
moderate_wet_mask = event_class == 'Moderate wet'
extreme_wet_mask = event_class == 'Extreme wet'
disturbance_mask = extreme_drought_mask | extreme_wet_mask
# Normal years used to estimate the undisturbed baseline lie within the
# historical interquartile range and are not recovery years immediately after
# an extreme event. climate_normal_mask retains its climatic meaning.
baseline_normal_mask = climate_normal_mask.copy()
for i, year in enumerate(year_labels):
    if year - 1 in historical_rainfall:
        previous_class = classify_year(historical_rainfall[year - 1])
        if previous_class in ('Extreme drought', 'Extreme wet'):
            baseline_normal_mask[i] = False

# Aliases retained for plotting; these masks contain only extreme events.
drought_mask = extreme_drought_mask
wet_mask = extreme_wet_mask
print(
    f"Reference period: {REFERENCE_START}-{REFERENCE_END}; "
    f"{len(reference_years)} complete years"
)
print(f"Simulation period: {n_real} years; Jan-Mar mean = {P_mean:.1f} mm")
print(
    f"Thresholds q10/q25/q75/q90 = "
    f"{q10:.2f}/{q25:.2f}/{q75:.2f}/{q90:.2f} mm\n"
)

# Export annual classifications so the event-selection procedure is auditable.
with (OUTPUT_DIR / 'precipitation_classification.csv').open(
        'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'year', 'jan_mar_precip_mm', 'event_class',
        'eligible_normal_baseline'
    ])
    for row in zip(year_labels, real_rainfall, event_class,
                   baseline_normal_mask):
        writer.writerow([row[0], f'{row[1]:.3f}', row[2], row[3]])

# ============================================================
# Plot style
# ============================================================
mpl.rcParams.update({
    # Use consistent typography for text, numbers, and mathematical notation.
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'font.size': 9,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ============================================================
# Model functions
# ============================================================
def equilibrium(H_eff, g, A, d, P=P_mean):
    """Return equilibrium seed-bank density under constant precipitation."""
    carryover = (1 - g) * (1 - d)
    return ((P * g * H_eff) / (1 - carryover) - P) / (A * g)

def simulate(N0, g, A, d, H_eff):
    """Simulate beginning-of-year seed-bank density from 1984 to 2013.

    Precipitation in year t updates N_t to N_{t+1}. The final precipitation
    value (2013) is displayed but is not used because N_2014 is not plotted.
    """
    N = np.zeros(n_real)
    N[0] = N0
    for t in range(n_real - 1):
        Pt = real_rainfall[t]
        N[t+1] = N[t] * ((Pt * g) / (Pt + A * g * N[t]) * H_eff
                         + (1 - g) * (1 - d))
    return N

def draw_precip_panel(ax):
    ax.bar(year_labels, real_rainfall, color='0.6', edgecolor='none', width=0.8)
    ax.axhline(y=P_mean, color='black', linestyle='--', linewidth=0.8,
               label=f'Mean ({P_mean:.0f} mm)')
    ax.axhline(y=q10, color='0.3', linestyle=':', linewidth=0.7)
    ax.axhline(y=q90, color='0.3', linestyle=':', linewidth=0.7,
               label=f'q10 / q90 ({q10:.1f} / {q90:.1f} mm)')
    for t in range(n_real):
        yr = year_labels[t]
        if drought_mask[t]:
            ax.axvspan(yr - 0.4, yr + 0.4, facecolor='0.8', edgecolor='none', alpha=0.7)
        elif wet_mask[t]:
            ax.axvspan(yr - 0.4, yr + 0.4, facecolor='0.9', edgecolor='none', alpha=0.5)
    ax.set_ylabel('Jan-Mar precip. (mm)')
    ax.set_xlim(1984, 2013)
    ax.legend(loc='upper right', fontsize=7, frameon=True, fancybox=False,
              edgecolor='0.3', framealpha=0.9)

# ============================================================
# Stability metrics
# ============================================================

# An isolated event lies outside q10/q90 and is preceded and followed by years
# within q25-q75. The upper range limit ensures that N_{t+2}, which is needed
# for the recovery metric, is available.
isolated_mask = np.zeros(n_real, dtype=bool)
for t in range(1, n_real - 2):
    if (climate_normal_mask[t-1] and disturbance_mask[t]
            and climate_normal_mask[t+1]):
        isolated_mask[t] = True

def resistance(N, P_arr):
    Yn_bar = N[1:][baseline_normal_mask[:-1]].mean()
    results = []
    for t in np.where(isolated_mask)[0]:
        Ye = N[t + 1]
        denom = abs(Ye - Yn_bar)
        Rt = Yn_bar / denom if denom > 1e-10 else np.inf
        results.append((t, P_arr[t], classify_year(P_arr[t]), Ye, Rt))
    return Yn_bar, results

def recovery(N, P_arr):
    Yn_bar = N[1:][baseline_normal_mask[:-1]].mean()
    results = []
    for t in np.where(isolated_mask)[0]:
        Ye = N[t + 1]
        Ye1 = N[t + 2]
        num = abs(Ye - Yn_bar)
        denom = abs(Ye1 - Yn_bar)
        Rs = num / denom if denom > 1e-10 else np.inf
        results.append((t, P_arr[t], classify_year(P_arr[t]), Ye, Ye1, Rs))
    return Yn_bar, results

# ============================================================
# One-at-a-time sensitivity analysis with species-informed parameter ranges
# ============================================================
# Parameter sources are the long-term species-year records and published seed
# survival estimates for MOBE, STMI, and EVMU. The species define biologically
# informed parameter ranges.
# g: mean germination fraction from long-term observations during 1990-2012.
# d = 1 - s_old, where s_old is winter survival multiplied by old-seed summer
# survival. H_eff = H_raw * s_new is effective seed production entering the
# following year's seed bank.
species_anchors = {
    'MOBE': {
        'g': 0.648424684,
        'winter_survival': 0.635,
        'old_seed_summer_survival': 0.430,
        'H_raw': 35.2,
        's_new': 0.102,
    },
    'STMI': {
        'g': 0.419136362,
        'winter_survival': 0.743,
        'old_seed_summer_survival': 0.616,
        'H_raw': 38.2,
        's_new': 0.145,
    },
    'EVMU': {
        'g': 0.088460635,
        'winter_survival': 0.916,
        'old_seed_summer_survival': 0.904,
        'H_raw': 112.7,
        's_new': 0.214,
    },
}

for traits in species_anchors.values():
    traits['d'] = 1.0 - (
        traits['winter_survival']
        * traits['old_seed_summer_survival']
    )
    traits['H_eff'] = traits['H_raw'] * traits['s_new']

H_base = species_anchors['STMI']['H_eff']
g_base = species_anchors['STMI']['g']
d_base = species_anchors['STMI']['d']
A_base = 4.0

def make_empirical_anchor_grid(low, middle, high):
    """Retain three anchors and insert two equally spaced intermediate values."""
    if not low < middle < high:
        raise ValueError('Empirical anchors must satisfy low < middle < high.')
    left = np.linspace(low, middle, 4)[:-1]
    right = np.linspace(middle, high, 4)
    return np.concatenate((left, right))

H_vals = make_empirical_anchor_grid(
    species_anchors['MOBE']['H_eff'],
    species_anchors['STMI']['H_eff'],
    species_anchors['EVMU']['H_eff'],
)

# One-at-a-time analysis of g with H_eff and d held constant
g_persistence_threshold = d_base / (H_base - 1.0 + d_base)
g_oat_lower = 0.15
if g_oat_lower <= g_persistence_threshold:
    raise ValueError('The lower g value must remain above the persistence threshold.')

g_vals = make_empirical_anchor_grid(
    g_oat_lower,
    species_anchors['STMI']['g'],
    species_anchors['MOBE']['g'],
)

d_vals = make_empirical_anchor_grid(
    species_anchors['EVMU']['d'],
    species_anchors['STMI']['d'],
    species_anchors['MOBE']['d'],
)

print('Species-informed sensitivity analysis: seven values per parameter.')
print('H_eff and d use species anchors; the lower g value ensures persistence.')
print(f'Baseline (STMI): H_eff={H_base:.3f}, '
      f'g={g_base:.3f}, d={d_base:.3f}, A={A_base:.1f}')
print(f'Positive-equilibrium threshold for g={g_persistence_threshold:.3f}; '
      f'lower sensitivity value={g_oat_lower:.2f}')

parameter_grids = {}
event_summary = {}

# ============================================================
# Panel (a): effective seed production H_eff
# ============================================================
print("=" * 60)
print("Panel (a): effective seed production H_eff")

N_H, lbl_H = [], []
event_rts_H, event_rss_H = [], []

for v in H_vals:
    N0 = equilibrium(v, g_base, A_base, d_base)
    N = simulate(N0, g_base, A_base, d_base, v)
    N_H.append(N)
    lbl_H.append(f'H_eff={v:g}')
    Ynb_r, rst = resistance(N, real_rainfall)
    Ynb_s, res = recovery(N, real_rainfall)
    rt_events = [r[-1] for r in rst]
    rs_events = [r[-1] for r in res]
    event_rts_H.append(rt_events)
    event_rss_H.append(rs_events)
    print(f"  H_eff={v:>5.2f}: N0={N0:>5.0f}, N_normal={Ynb_r:>5.0f}, "
          f"Event Rt={np.round(rt_events, 3)}, Event Rs={np.round(rs_events, 3)}")

parameter_grids['H'] = H_vals
event_summary['H'] = (np.asarray(event_rts_H), np.asarray(event_rss_H))


# ============================================================
# Panel (b): germination fraction g
# ============================================================
print("\n" + "=" * 60)
print("Panel (b): germination fraction g")

N_g, lbl_g = [], []
event_rts_g, event_rss_g = [], []

for v in g_vals:
    N0 = equilibrium(H_base, v, A_base, d_base)
    N = simulate(N0, v, A_base, d_base, H_base)
    N_g.append(N)
    lbl_g.append(f'g={v:g}')
    Ynb_r, rst = resistance(N, real_rainfall)
    Ynb_s, res = recovery(N, real_rainfall)
    rt_events = [r[-1] for r in rst]
    rs_events = [r[-1] for r in res]
    event_rts_g.append(rt_events)
    event_rss_g.append(rs_events)
    print(f"  g={v:.2f}: N0={N0:>5.0f}, N_normal={Ynb_r:>5.0f}, "
          f"Event Rt={np.round(rt_events, 3)}, Event Rs={np.round(rs_events, 3)}")

parameter_grids['g'] = g_vals
event_summary['g'] = (np.asarray(event_rts_g), np.asarray(event_rss_g))


# ============================================================
# Panel (c): dormant-seed mortality rate d
# ============================================================
print("\n" + "=" * 60)
print("Panel (c): dormant-seed mortality rate d")

N_d, lbl_d = [], []
event_rts_d, event_rss_d = [], []

for v in d_vals:
    N0 = equilibrium(H_base, g_base, A_base, v)
    N = simulate(N0, g_base, A_base, v, H_base)
    N_d.append(N)
    lbl_d.append(f'd={v:g}')
    Ynb_r, rst = resistance(N, real_rainfall)
    Ynb_s, res = recovery(N, real_rainfall)
    rt_events = [r[-1] for r in rst]
    rs_events = [r[-1] for r in res]
    event_rts_d.append(rt_events)
    event_rss_d.append(rs_events)
    print(f"  d={v:.2f}: N0={N0:>5.0f}, N_normal={Ynb_r:>5.0f}, "
          f"Event Rt={np.round(rt_events, 3)}, Event Rs={np.round(rs_events, 3)}")

parameter_grids['d'] = d_vals
event_summary['d'] = (np.asarray(event_rts_d), np.asarray(event_rss_d))



keys = ['H', 'g', 'd']
parameter_titles = ['Effective seed yield', 'Germination fraction',
                    'Seed mortality']
xlabels = [r'$H_i$', r'$g_i$', r'$d_i$']
base_values = {'H': H_base, 'g': g_base, 'd': d_base}

event_indices = np.where(isolated_mask)[0]
event_labels = []
for t in event_indices:
    event_labels.append(
        f'{year_labels[t]} {classify_year(real_rainfall[t]).lower()} '
        f'({real_rainfall[t]:.1f} mm)'
    )

event_styles = [
    dict(color='black', linestyle='-',  marker='o'),
    dict(color='0.35', linestyle=':', marker='^'),
    dict(color='black', linestyle='--', marker='s'),
]
if len(event_labels) == 0:
    raise ValueError(
        'No isolated normal-extreme-normal events were found during the '
        'simulation period.'
    )
if len(event_labels) > len(event_styles):
    raise ValueError(
        'The number of events exceeds the available plotting styles.'
    )

fig_sens, axes_sens = plt.subplots(
    2, 3, figsize=(7.2, 4.6), sharey='row'
)
top_letters = ['(a)', '(b)', '(c)']
bottom_letters = ['(d)', '(e)', '(f)']

for col, (key, title, xlbl) in enumerate(
        zip(keys, parameter_titles, xlabels)):
    vals = np.asarray(parameter_grids[key], dtype=float)
    event_rt, event_rs = event_summary[key]

    for event_no, (style, event_label) in enumerate(
            zip(event_styles, event_labels)):
        common_style = dict(
            linewidth=1.15,
            markersize=3.3,
            markerfacecolor='white',
            markeredgewidth=0.7,
        )
        axes_sens[0, col].plot(
            vals, event_rt[:, event_no], label=event_label,
            **style, **common_style
        )
        axes_sens[1, col].plot(
            vals, event_rs[:, event_no], label=event_label,
            **style, **common_style
        )

    for row in range(2):
        ax = axes_sens[row, col]
        ax.axvline(base_values[key], color='0.55', linestyle='--',
                   linewidth=0.75, zorder=0)
        ax.tick_params(direction='in')
        ax.margins(x=0.04)

    axes_sens[0, col].set_title(
        f'{top_letters[col]}  {title}', loc='left', fontsize=8.5, pad=3
    )
    axes_sens[1, col].text(
        0.0, 1.03, bottom_letters[col], transform=axes_sens[1, col].transAxes,
        ha='left', va='bottom', fontsize=8.5, fontweight='bold'
    )
    axes_sens[1, col].set_xlabel(xlbl)
    axes_sens[0, col].set_yscale('log')
    axes_sens[0, col].set_ylim(0.8, 2.5)
    axes_sens[0, col].yaxis.set_major_locator(
        FixedLocator([0.8, 1.0, 1.5, 2.0, 2.5])
    )
    axes_sens[0, col].yaxis.set_major_formatter(
        FuncFormatter(lambda value, position: f'{value:g}')
    )
    axes_sens[0, col].yaxis.set_minor_locator(NullLocator())
    axes_sens[0, col].yaxis.set_minor_formatter(NullFormatter())
    axes_sens[0, col].tick_params(axis='y', which='major', left=True, length=3, width=0.8)
    axes_sens[0, col].tick_params(
        axis='y', which='minor', left=True, right=False, length=2, width=0.6
    )
    axes_sens[1, col].set_ylim(0.8, 2.4)
    axes_sens[1, col].yaxis.set_major_locator(FixedLocator([0.8, 1.2, 1.6, 2.0, 2.4]))
    axes_sens[1, col].yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    axes_sens[1, col].yaxis.set_minor_locator(NullLocator())

    # Five ticks including endpoints, with matching x axes in each column.
    x_limits = (0.0, 28.0) if key == 'H' else (0.0, 0.8)
    for row in range(2):
        ax = axes_sens[row, col]
        ax.set_xlim(*x_limits)
        ax.xaxis.set_major_locator(FixedLocator(np.linspace(*x_limits, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis='both', which='major', length=3, width=0.8)

axes_sens[0, 0].set_ylabel(r'Resistance, $\Omega$')
axes_sens[1, 0].set_ylabel(r'Recovery, $\Delta$')

legend_handles, legend_labels = axes_sens[0, 0].get_legend_handles_labels()
axes_sens[0, 0].legend(
    legend_handles, legend_labels,
    loc='upper right', ncol=1, frameon=False, fontsize=6.8,
    handlelength=2.4, labelspacing=0.45, borderaxespad=0.55
)

fig_sens.align_ylabels()
fig_sens.tight_layout(pad=0.7, w_pad=0.7, h_pad=0.8)
fig_sens.savefig(OUTPUT_DIR / 'figure2_sensitivity_analysis.pdf',
                 bbox_inches='tight')
fig_sens.savefig(OUTPUT_DIR / 'figure2_sensitivity_analysis.svg',
                 format='svg', bbox_inches='tight')
fig_sens.savefig(OUTPUT_DIR / 'figure2_sensitivity_analysis.png',
                 dpi=600, bbox_inches='tight')

# ============================================================
# Three-species comparison: MOBE, STMI, and EVMU
# ============================================================
print("\n" + "=" * 60)
print("Three-species comparison: resistance and recovery")

species = [
    ("MOBE", 0.648, 0.727, 3.5904, 4.0),
    ("STMI", 0.419, 0.542, 5.5390, 4.0),
    ("EVMU", 0.088, 0.172, 24.1178, 3.0),
]

N_species = {}
N0_species = {}
all_rst, all_res = {}, {}

for name, g, d, H_eff, A in species:
    N0 = equilibrium(H_eff, g, A, d)
    N_species[name] = simulate(N0, g, A, d, H_eff)
    N0_species[name] = N0
    Ynb_r, rst = resistance(N_species[name], real_rainfall)
    Ynb_s, res = recovery(N_species[name], real_rainfall)
    all_rst[name] = (Ynb_r, rst)
    all_res[name] = (Ynb_s, res)
    avg_rt = np.mean([r[-1] for r in rst]) if rst else float('nan')
    avg_rs = np.mean([r[-1] for r in res]) if res else float('nan')
    print(f"{name}: g={g}, d={d}, H_eff={H_eff}, A={A}; N0={N0:.1f}, "
          f"N_normal={Ynb_r:.1f}, Mean Rt={avg_rt:.3f}, Mean Rs={avg_rs:.3f}")

# Plot the three species
linestyles = ['-', '--', '-.']
fig_sp, ax_sp = plt.subplots(figsize=(6.3, 4.5))

for t in range(n_real):
    yr = year_labels[t]
    if drought_mask[t]:
        ax_sp.axvspan(yr - 0.4, yr + 0.4, facecolor='0.8', edgecolor='none', alpha=0.7)
    elif wet_mask[t]:
        ax_sp.axvspan(yr - 0.4, yr + 0.4, facecolor='0.9', edgecolor='none', alpha=0.5)

for idx, (name, g, d, H_eff, A) in enumerate(species):
    ax_sp.plot(year_labels, N_species[name], color='black',
               linestyle=linestyles[idx], linewidth=1.8,
               label=f'{name} ($g={g},\\,d={d},\\,H={H_eff}$)')

ax_sp.set_xlabel('Year')
ax_sp.set_ylabel(r'Abundance  $N_{i,t}$')
ax_sp.set_xlim(1984, 2013)
ax_sp.set_ylim(bottom=0)
ax_sp.legend(loc='upper right', frameon=True, fancybox=False,
             edgecolor='0.3', framealpha=0.9,
             handlelength=2.5, handleheight=0.7, fontsize=7)
fig_sp.tight_layout(pad=0.5)
fig_sp.savefig(OUTPUT_DIR / 'three_species_comparison.svg', format='svg')
fig_sp.savefig(OUTPUT_DIR / 'three_species_comparison.png', dpi=600)

# ============================================================
# Precipitation sequence
# ============================================================
fig_precip, ax_precip = plt.subplots(figsize=(6.3, 3.0))
draw_precip_panel(ax_precip)
fig_precip.tight_layout(pad=0.5)
fig_precip.savefig(OUTPUT_DIR / 'precipitation_sequence.svg', format='svg')
fig_precip.savefig(OUTPUT_DIR / 'precipitation_sequence.png', dpi=600)

plt.close('all')

# Console summary
print("\n" + "=" * 60)
print(f'Precipitation: DESST Jan-Mar, {TARGET_START}-{TARGET_END} '
      f'({n_real} years)')
print(f'Reference period: {REFERENCE_START}-{REFERENCE_END}; '
      f'{len(reference_years)} complete years')
print(f'Mean precipitation during the simulation period: {P_mean:.1f} mm')
print(f'Extreme drought (<q10={q10:.2f}): '
      f'{np.sum(extreme_drought_mask)} years')
print(f'Moderate drought (q10-q25): {np.sum(moderate_drought_mask)} years')
print(f'Normal (q25-q75): {np.sum(climate_normal_mask)} years')
print(f'Moderately wet (q75-q90): {np.sum(moderate_wet_mask)} years')
print(f'Extremely wet (>q90={q90:.2f}): '
      f'{np.sum(extreme_wet_mask)} years')
print('Isolated normal-extreme-normal event years:',
      year_labels[isolated_mask].tolist())
print('=' * 60)


