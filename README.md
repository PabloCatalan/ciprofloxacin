# A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions

Data and code to reproduce the results and figures of the manuscript

**"A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions"**
by Marina de la Fuente García, Javier Molina-Hernández, Pablo Chávez, Pablo Catalán and Saúl Ares.

## Overview

We model the action of ciprofloxacin (CIP) on *E. coli* with two coarse-grained variables, the cell density $N$ and the population's replication-fork content $f$. The fork density $\varphi = f/N$ obeys an equation that factorises into a function of $N$ times a function of $\varphi$, so the manifold $\varphi = \varphi^*(c)$ is invariant and attracting, and on it the system reduces **exactly** to a logistic equation

$$\dot N = r(c)\,N\left(1 - \frac{N}{K(c)}\right),\qquad r(c) = r_0\,\Phi(c)\,\frac{1 - qc\,h(c)}{h(c)},\qquad K(c) = K_0\,\bigl(1 - qc\,h(c)\bigr),$$

with $h(c) = 1 + (c/c_0)^\alpha$ and $\Phi(c) = \varphi^*(c)/\varphi^*(0)$. Fixing the fork completion rate at $d = 1\ \mathrm{h^{-1}}$ leaves five parameters, $(r_0, K_0, q, c_0, \alpha)$, which are fitted by Bayesian inference simultaneously to all growth curves of a CIP dose-response experiment. The MIC follows analytically as the root of $q\,c^*\,h(c^*) = 1$.

## Repository structure

| Path | Content |
|---|---|
| `functions.py` | Model equations, data loading, Bayesian fits, MIC calculation, posterior summaries and shared plotting code. Imported by all notebooks. |
| `cip_model_bayesian_inference.ipynb` | Runs every fit and writes all intermediate results to `results/`. |
| `main_figures.ipynb` | Figures 1–5 of the main text. |
| `supp_figures.ipynb` | Supplementary Figures S1–S7. |
| `preprocessing_median.py` | Builds `data/cip_dose_response_all.csv` (rolling-median smoothed OD) from `data/cip_dose_response_raw.csv`. Included for completeness; the smoothed file is already provided. |
| `data/` | Raw and smoothed OD600 data. |
| `results/` | Posteriors (`.nc`, NetCDF) and summary tables (`.csv`) written by the inference notebook. |
| `figures/` | PDF figures written by the figure notebooks. |

### Figures and the results they use

| Figure | Notebook | Results files |
|---|---|---|
| Fig. 1, Figs. S2–S5 | `main_figures`, `supp_figures` | `cip_model_standard_{rep}_12h.nc`, `cip_logistic_standard_{rep}_dose_{c}_12h.nc`, `rK_{rep}_12h.csv` |
| Fig. 2 | `main_figures` | `mic_convergence_data_linear_add.csv`, `cip_model_cv.csv` |
| Figs. 3–4, Table 1 | `main_figures` | `cip_model_{medium}_{cond}_12h.nc`, `parameters_table.csv` |
| Fig. 5 | `main_figures` | `parameters_table.csv`, `{TMP,CHL}_linear_fit.nc` |
| Fig. S1 | `supp_figures` | `parameters_table.csv` (glucose 0.2%) |
| Fig. S6 | `supp_figures` | `mic_convergence_combinations_12h.csv`, `rhat_scores_combinations_12h.csv` |
| Fig. S7 | `supp_figures` | `parameters_table_fixed_alpha.csv`, `{TMP,CHL}_linear_fit_fixed_alpha.nc` |

## Installation

```bash
git clone https://github.com/PabloCatalan/ciprofloxacin.git
cd ciprofloxacin
conda env create -f environment.yml
conda activate cip_model
```

## Usage

Run the notebooks from the repository root, in this order:

1. **`cip_model_bayesian_inference.ipynb`** fits the model and writes everything in `results/`.
2. **`main_figures.ipynb`** and **`supp_figures.ipynb`** read `results/` and write the PDFs to `figures/`.

Each fit is skipped when its output file already exists in `results/`, and the stored posterior is loaded instead. With the posteriors in place, the inference notebook therefore only rebuilds the summary tables and runs in minutes. To repeat a fit, delete its `.nc` file (or call the fitting function with `refit=True`). Fitting everything from scratch takes several hours on a 4-core machine (about 150 fits). Sampling and the posterior draws used in plots are seeded (`RANDOM_SEED` in `functions.py`).

### Notes on the data

- All fits use the first 12 h of each growth curve (`T_MAX` in `functions.py`).
- Two rows (F and G) of the chloramphenicol plate (day 251203) failed to grow in every well, including the drug-free controls. They are excluded from all fits and plots by `load_data()` (`EXCLUDED_ROWS` in `functions.py`).
- The standard-medium experiments were inoculated at OD ≈ 0.001 and the medium-perturbation series (glucose, glycerol, TMP, CHL) at OD ≈ 0.003.

### Units

Raw data, all calculations and every file in `results/` use **µg/mL** for CIP. Figures and the manuscript report **ng/mL**: a stored value of `0.015` is 15 ng/mL. The conversion happens only at the plotting stage.

## Citation

> de la Fuente García M, Molina-Hernández J, Chávez P, Catalán P, Ares S. A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions. *Journal*, year. DOI: to be added.

## License

MIT License. See `LICENSE`.
