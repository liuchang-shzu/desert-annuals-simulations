# desert-annuals-simulations
# Seed-Bank Dynamics under Precipitation Variability

This repository contains the data and Python code used to reproduce the
soil seed-bank simulations and sensitivity analyses presented in the
manuscript:

> Water-Use Trade-Offs and Seed-Bank Dynamics in Dryland Annual Plants:
> A Fitness Framework for Species Diversity

## Repository Contents

- `seedbank.py` classifies precipitation years, simulates soil seed-bank
  dynamics for three Sonoran Desert winter annual species, and generates
  Figure 1.
- `sensitivity.py` performs one-at-a-time sensitivity analyses of effective
  seed production, germination fraction, and dormant-seed mortality, and
  generates Figure 2.
- `data/precip1922_2017.csv` contains the precipitation records used in the
  analyses.
- `outputs/` is created automatically when the scripts are run.

## Data Source

The precipitation data were obtained from the Santa Rita Experimental Range
precipitation database. The analyses use cumulative January-March
precipitation recorded at the DESST station.

The source data record precipitation in hundredths of an inch. Values are
converted to millimetres in the scripts using a conversion factor of 0.254.

Historical precipitation thresholds are calculated from 53 complete
January-March seasons between 1965 and 2017. The seed-bank simulations cover
the period from 1984 to 2013.

The DESST precipitation sequence is used as an independent regional
environmental forcing. The simulations represent theoretical responses of
parameterized species to a common precipitation sequence and are not
reconstructions of population dynamics at the original demographic study
site.

## Species Parameters

The simulations include the following Sonoran Desert winter annual species:

- `MOBE`: *Monoptilon bellioides*
- `STMI`: *Stylocline micropoides*
- `EVMU`: *Evax multicaulis*

Species-specific germination fractions and seed-survival parameters were
derived from the published sources cited in the manuscript.

In the model, `H_i` denotes effective seed production after accounting for
new-seed survival. The values of `A_i` are illustrative model parameters rather
than measurements of individual water consumption.

## Temporal Indexing

`N_t` denotes seed-bank density at the beginning of year `t`. Precipitation in
year `t` determines seed-bank density in year `t + 1`.

Consequently, precipitation from 1984 through 2012 determines the plotted
seed-bank states from 1985 through 2013. Precipitation in 2013 would determine
the 2014 seed-bank state, which is outside the plotted period.

## Requirements

The scripts require Python 3 and the following packages:

- NumPy
- Matplotlib

Install the required packages with:

```bash
pip install numpy matplotlib
```

## Running the Analyses

Run the seed-bank simulation with:

```bash
python seedbank.py
```

This script generates the following Figure 1 files in the automatically
created `outputs/` directory:

- `figure1_seedbank_dynamics.pdf`
- `figure1_seedbank_dynamics.svg`
- `figure1_seedbank_dynamics.png`

Run the sensitivity analysis with:

```bash
python sensitivity.py
```

This script generates the following principal Figure 2 files:

- `figure2_sensitivity_analysis.pdf`
- `figure2_sensitivity_analysis.svg`
- `figure2_sensitivity_analysis.png`

It also exports `precipitation_classification.csv` and additional diagnostic
figures to the `outputs/` directory.

## Precipitation Classification

January-March precipitation is classified using the 10th, 25th, 75th, and
90th percentiles of the historical reference period:

- Below the 10th percentile: extremely dry
- From the 10th to below the 25th percentile: moderately dry
- From the 25th through the 75th percentile: normal
- Above the 75th through the 90th percentile: moderately wet
- Above the 90th percentile: extremely wet

Resistance and recovery are evaluated for isolated extreme events that are
preceded and followed by normal precipitation years.

## Reproducibility

Both scripts use paths relative to the repository directory. No changes to
local file paths are required when the repository structure is retained.

Generated files are written to the `outputs/` directory. The generated figures
do not need to be stored in the repository because they can be reproduced by
running the scripts.
