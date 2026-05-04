"""
functions.py
============
Shared utilities for Bayesian inference, model simulation, and plotting used
throughout the ciprofloxacin dose-response analysis.

All functions are imported via `from functions import *` in the three analysis
notebooks:
  - cip_model_bayesian_inference.ipynb  (inference + intermediate results)
  - main_figures.ipynb                  (Figs 1–4)
  - supp_figures.ipynb                  (Figs S1–S6)

References
----------
Model derivation: see Section 2.1 of the manuscript (Eq. 3–5).
Bayesian inference details: Materials and Methods section.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import pylab
import numpy as np
from scipy.integrate import solve_ivp
import itertools
import subprocess
import pandas as pd
import os
from scipy.optimize import curve_fit, root_scalar
from scipy.stats import median_abs_deviation
import pymc as pm
import arviz as az

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['font.sans-serif'] = "Arial"

# Fixed random seed for reproducibility of any stochastic plotting operations
# (e.g. random draws from the posterior used in posterior-predictive plots).
RANDOM_SEED = 89271458
rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Model equations
# ---------------------------------------------------------------------------

def g(c, t, y0, r0, K0, q, c0, alpha):
    """
    Closed-form solution of the CIP logistic model (manuscript Eq. 5).

    The model collapses the two-ODE fork/cell system into a single logistic
    equation via quasi-steady-state reduction (Eq. 3).  The CIP-dependent
    growth rate and carrying capacity are:

        r(c) = r0 / (1 + (c/c0)^alpha) - q*c          (Eq. 4, left)
        K(c) = K0 * (1 - q*c*(1 + (c/c0)^alpha) / r0)  (Eq. 4, right)

    The logistic ODE dN/dt = r(c)*N*(1 - N/K(c)) has the explicit solution
    returned here (Eq. 5).

    Parameters
    ----------
    c : float or array-like
        Ciprofloxacin concentration (µg/mL).
    t : float or array-like
        Time (hours).
    y0 : float
        Initial cell density (OD_600).
    r0 : float
        Antibiotic-free growth rate (h^-1).
    K0 : float
        Antibiotic-free carrying capacity (OD units).
    q : float
        CIP killing efficiency (mL/µg/h); controls the linear death term.
    c0 : float
        Half-saturation concentration for fork stalling (µg/mL); the
        concentration at which the fork-initiation rate is halved.
    alpha : float
        Hill coefficient encoding cooperativity of CIP–gyrase binding.

    Returns
    -------
    float or ndarray
        Predicted OD_600 at time t for concentration c.
    """
    # Small epsilon prevents division by zero if r0 or c0 are exactly 0
    eps = 1e-6
    r0_safe = r0 + eps
    c0_safe = c0 + eps

    # Concentration-dependent carrying capacity K(c)
    K = K0 - K0 * q * c * (1 + (c / c0_safe) ** alpha) / r0_safe

    # Concentration-dependent net growth rate r(c):
    # the Hill term captures fork stalling (bacteriostatic effect);
    # the q*c term is direct lethal killing (bactericidal effect).
    r = r0 / (1 + (c / c0_safe) ** alpha) - q * c

    # Standard logistic closed-form solution N(t)
    return (K * y0) / (y0 + (K - y0) * np.exp(-r * t))


def logistic(t, y0, r, K):
    """
    Standard (antibiotic-free) logistic growth curve.

    Used as the baseline comparison model: each CIP concentration is fitted
    independently with its own (r, K) pair, without any mechanistic
    constraint linking curves across concentrations.  This is the "logistic"
    reference in Fig. 1B and panels C–D.

    Parameters
    ----------
    t : array-like
        Time (hours).
    y0 : float
        Initial OD_600.
    r : float
        Growth rate (h^-1).
    K : float
        Carrying capacity (OD units).

    Returns
    -------
    ndarray
        OD_600 at each time point.
    """
    return (K * y0) / (y0 + (K - y0) * np.exp(-r * t))


# ---------------------------------------------------------------------------
# Bayesian inference
# ---------------------------------------------------------------------------

def fit_model(DF, filename, y0_input, draws=1000):
    """
    Fit the five-parameter CIP logistic model to a single experimental condition
    using Bayesian MCMC inference (PyMC + NUTS sampler).

    All CIP dose-response curves from the provided DataFrame are fitted
    *simultaneously* with a single shared parameter set (r0, K0, q, c0, alpha),
    which is what constrains the model across concentrations and enables
    robust MIC estimation (see Section 2.2 and Materials & Methods).

    The posterior is stored as a NetCDF file for later loading with ArviZ.

    Parameters
    ----------
    DF : pd.DataFrame
        Tidy DataFrame for one experimental replicate / medium condition.
        Must contain columns: 'dosisAb' (CIP concentration), 'Hours' (time),
        'ODsmooth' (rolling-median-smoothed OD_600).
    filename : str
        Path to the output NetCDF file (e.g. 'results/cip_model_standard_1_12h.nc').
    y0_input : float
        Fixed initial OD_600 (N0 in the model); set to the inoculum density.
    draws : int
        Number of posterior samples per chain (default 1000).
        The tuning phase always uses 2000 steps.

    Returns
    -------
    arviz.InferenceData
        Posterior samples, also saved to `filename`.

    Notes
    -----
    Priors (see Materials & Methods):
      r0    ~ HalfNormal(sigma=1.0)    — positive growth rate
      K0    ~ HalfNormal(sigma=1.0)    — positive carrying capacity
      q     ~ HalfNormal(sigma=10)     — positive killing rate; wide prior
      c0    ~ LogNormal(log(0.01), 1)  — strictly positive; log-scale prior
                                         handles the small-number regime well
      alpha ~ HalfNormal(sigma=4)      — positive Hill coefficient
      sigma ~ HalfNormal(sigma=0.1)    — observational noise on OD
    """
    with pm.Model() as generative_model:

        # Register CIP concentration, time, and initial density as PyMC Data
        # objects so they can be swapped for posterior-predictive checks later.
        c  = pm.Data("c",  DF.dosisAb)
        t  = pm.Data("t",  DF.Hours)
        y0 = pm.Data("y0", y0_input)

        # --- Priors ---
        r0    = pm.HalfNormal('r0')                         # antibiotic-free growth rate (h^-1)
        K0    = pm.HalfNormal('K0')                         # antibiotic-free carrying capacity
        q     = pm.HalfNormal('q', sigma=10.0)              # killing efficiency; wide prior
        c0    = pm.LogNormal('c0', mu=np.log(0.01), sigma=1.0)  # half-saturation concentration
        alpha = pm.HalfNormal('alpha', sigma=4.0)           # Hill cooperativity coefficient
        sigma = pm.HalfNormal('sigma', sigma=0.1)           # OD measurement noise

        # --- Likelihood ---
        # The model prediction g(c, t, y0, ...) is compared to smoothed OD data.
        # A Normal likelihood assumes additive Gaussian noise on the OD readings.
        likelihood = pm.Normal(
            "OD",
            mu=g(c, t, y0, r0, K0, q, c0, alpha),
            sigma=sigma,
            observed=DF.ODsmooth
        )

        # --- Sampling ---
        # NUTS (No-U-Turn Sampler) with 4 parallel chains.
        # target_accept=0.95 reduces divergences in the curved posteriors
        # that arise from near-MIC concentrations.
        idata = pm.sample(
            draws=draws,
            tune=2000,
            target_accept=0.95,
            cores=4,
            return_inferencedata=True
        )

        # Save only the groups needed for downstream analysis to keep file sizes manageable.
        safe_groups = ["posterior", "posterior_predictive", "sample_stats"]
        idata.to_netcdf(filename, groups=safe_groups)

    return idata


def fit_model_fixed_alpha(DF, filename, y0_input, draws=1000):
    """
    Variant of fit_model() with alpha constrained to a narrow Normal prior.

    Used in the sensitivity analysis of Supplementary Fig. S6: alpha is
    effectively fixed at 6.21 (the mean across all freely-fitted conditions)
    to test whether the r0–q correlation and MIC dichotomy survive when the
    Hill coefficient is not a free parameter.

    As described in the manuscript Discussion, this regularisation helps in
    most conditions but produces numerical instabilities at very high CHL
    doses, making the free-alpha approach preferable in general.

    Parameters
    ----------
    DF : pd.DataFrame
        Same format as fit_model().
    filename : str
        Output NetCDF path.
    y0_input : float
        Initial OD_600.
    draws : int
        Number of posterior samples per chain (default 1000).
        The tuning phase always uses 2000 steps.

    Returns
    -------
    arviz.InferenceData
    """
    with pm.Model() as generative_model:

        c  = pm.Data("c",  DF.dosisAb)
        t  = pm.Data("t",  DF.Hours)
        y0 = pm.Data("y0", y0_input)

        r0    = pm.HalfNormal('r0')
        K0    = pm.HalfNormal('K0')
        q     = pm.HalfNormal('q', sigma=10.0)
        c0    = pm.LogNormal('c0', mu=np.log(0.01), sigma=1.0)

        # Tightly constrained Normal around the global mean alpha (6.21).
        # sigma=0.01 makes this effectively a fixed value while remaining
        # technically differentiable for the gradient-based NUTS sampler.
        alpha = pm.Normal('alpha', mu=6.21, sigma=0.01)

        sigma = pm.HalfNormal('sigma', sigma=0.1)

        likelihood = pm.Normal(
            "OD",
            mu=g(c, t, y0, r0, K0, q, c0, alpha),
            sigma=sigma,
            observed=DF.ODsmooth
        )

        idata = pm.sample(
            draws=draws,
            tune=2000,
            target_accept=0.95,
            cores=4,
            return_inferencedata=True
        )

        safe_groups = ["posterior", "posterior_predictive", "sample_stats"]
        idata.to_netcdf(filename, groups=safe_groups)

    return idata


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_data(DF, ax):
    """
    Plot smoothed OD_600 growth curves from experimental data.

    For each CIP concentration, the median OD across wells/replicates is
    shown as a solid line and the interquartile range (Q1–Q3) as a shaded
    band.  Colors follow the 'inferno' colormap ordered by concentration.

    Parameters
    ----------
    DF : pd.DataFrame
        Must contain: 'well', 'rep', 'Minutes', 'Hours', 'dosisAb', 'ODsmooth'.
    ax : matplotlib.axes.Axes
        Target axes.

    Returns
    -------
    dict
        {label: handle} mapping suitable for passing to ax.legend().
    """
    DF = DF.sort_values(by=['well', 'rep', 'Minutes'])
    unique_doses = sorted(DF['dosisAb'].unique())

    # Assign a distinct inferno color to each CIP concentration
    colors = plt.get_cmap('inferno', len(unique_doses))
    color_dict = {dose: colors(i) for i, dose in enumerate(unique_doses)}

    # Aggregate: median + IQR across wells and replicates for each (dose, time) pair
    stats_df = DF.groupby(['dosisAb', 'Minutes'])['ODsmooth'].agg(
        OD_median='median',
        OD_q1=lambda x: x.quantile(0.25),
        OD_q3=lambda x: x.quantile(0.75)
    ).reset_index()
    stats_df['Hours'] = stats_df['Minutes'] / 60

    for dose in stats_df['dosisAb'].unique():
        subset = stats_df[stats_df['dosisAb'] == dose]
        # Label in ng/mL (data are stored in µg/mL)
        line, = ax.plot(subset['Hours'], subset['OD_median'],
                        label=1000 * dose, color=color_dict[dose])
        ax.fill_between(
            subset['Hours'],
            subset['OD_q1'],
            subset['OD_q3'],
            color=line.get_color(),
            alpha=0.2
        )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))   # deduplicate legend entries

    ax.set_xlabel('time (hours)')
    ax.set_ylabel('OD$_{600}$')
    return by_label


def plot_pred(ax, filename, unique_doses, y0):
    """
    Overlay posterior-predictive model trajectories on a data plot.

    For each CIP concentration, 1000 parameter vectors are drawn at random
    from the stored posterior, the model is evaluated for each, and the
    mean ± 1 SD envelope is plotted as a dashed line with a shaded band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to overlay predictions (should already contain data).
    filename : str
        Path to the NetCDF file with the fitted posterior.
    unique_doses : array-like
        Sorted list of CIP concentrations (µg/mL) to predict.
    y0 : float
        Initial OD_600 used during fitting.
    """
    # 'inferno' colormap must match the one used in plot_data()
    colors = plt.get_cmap('inferno', len(unique_doses))
    color_dict = {dose: colors(i) for i, dose in enumerate(unique_doses)}

    T = np.linspace(0, 12, 100)   # dense time grid for smooth curves

    # Load posterior and flatten chain/draw dimensions into a single 'samples' axis
    samples = az.from_netcdf(filename)
    posterior = samples.posterior.stack(samples=("draw", "chain"))

    for c in unique_doses:
        all_sims = []
        # Draw 1000 random posterior samples (with replacement) for Monte Carlo averaging
        for i in np.random.randint(0, posterior.samples.size, 1000):
            r0i    = posterior["r0"].values[i]
            K0i    = posterior["K0"].values[i]
            qi     = posterior["q"].values[i]
            c0i    = posterior["c0"].values[i]
            alphai = posterior["alpha"].values[i]
            ypred  = g(c, T, y0, r0i, K0i, qi, c0i, alphai)
            all_sims.append(ypred)

        average_sim = np.mean(all_sims, axis=0)
        std_sim     = np.std(all_sims, axis=0)

        ax.plot(T, average_sim, ls='--', color=color_dict[c])
        ax.fill_between(T, average_sim - std_sim, average_sim + std_sim,
                        color=color_dict[c], alpha=0.25)


def plot_pred_logistic(rep, ax, unique_doses, y0):
    """
    Overlay posterior-predictive *independent logistic* trajectories.

    Unlike plot_pred(), each CIP concentration has its own separately fitted
    (r, K) posterior stored in individual NetCDF files
    ('results/cip_logistic_standard_{rep}_dose_{c}_12h.nc').
    This is the comparison model shown in Fig. 1B / Figs. S2–S5 panel B.

    Parameters
    ----------
    rep : int
        Biological replicate index (1–5).
    ax : matplotlib.axes.Axes
    unique_doses : array-like
        CIP concentrations (µg/mL).
    y0 : float
        Initial OD_600.
    """
    colors = plt.get_cmap('inferno', len(unique_doses))
    color_dict = {dose: colors(i) for i, dose in enumerate(unique_doses)}

    T = np.linspace(0, 12, 100)

    for c in unique_doses:
        all_sims = []
        # Each concentration has its own NetCDF file from the per-curve fit
        samples  = az.from_netcdf(f"results/cip_logistic_standard_{rep}_dose_{c}_12h.nc")
        posterior = samples.posterior.stack(samples=("draw", "chain"))

        for i in np.random.randint(0, posterior.samples.size, 1000):
            Ki    = posterior["K"].values[i]
            ri    = posterior["r"].values[i]
            ypred = logistic(T, y0, ri, Ki)
            all_sims.append(ypred)

        average_sim = np.mean(all_sims, axis=0)
        std_sim     = np.std(all_sims, axis=0)

        ax.plot(T, average_sim, ls='--', color=color_dict[c])
        ax.fill_between(T, average_sim - std_sim, average_sim + std_sim,
                        color=color_dict[c], alpha=0.25)


# ---------------------------------------------------------------------------
# MIC calculation
# ---------------------------------------------------------------------------

def solve_mic(r0, q, c0, alpha):
    """
    Numerically solve for the MIC (c*) given a single parameter vector.

    The MIC satisfies the implicit equation (manuscript Eq. 6):

        q * c* * (1 + (c*/c0)^alpha) = r0

    which is obtained by setting r(c) = 0 (equivalently K(c) = 0).
    The root is found with Brent's method on the bracket [0, 1000] µg/mL.

    Parameters
    ----------
    r0, q, c0, alpha : float
        Model parameters for a single posterior sample.

    Returns
    -------
    float
        MIC estimate in µg/mL, or np.nan if the solver fails to bracket a root.
    """

    # Note: Returns MIC in µg/mL. Multiply by 1000 when plotting for the manuscript (ng/mL).
    
    def equation(MIC):
        # Rewritten as f(MIC) = 0 for root_scalar
        return q * MIC * (1 + (MIC / c0) ** alpha) - r0

    try:
        res = root_scalar(equation, bracket=[0, 1000], method='brentq')
        return res.root
    except ValueError:
        # No root in [0, 1000]: can happen if parameters place c* outside range
        return np.nan


def get_mic(idata_rep, output='simple'):
    """
    Compute the posterior distribution of the MIC from an InferenceData object.

    The MIC is derived analytically from the five-parameter posterior for each
    MCMC sample (Eq. 6), yielding a full posterior distribution over the MIC.
    This propagates all parameter uncertainty into the MIC estimate, making it
    more robust than computing it from posterior means alone.

    Parameters
    ----------
    idata_rep : arviz.InferenceData
        Fitted posterior (loaded from NetCDF).
    output : {'simple', 'full'}
        'simple' — return (mean, std) of the MIC posterior (default).
        'full'   — return the full array of MIC posterior samples (used for
                   Fig. 2A, where the full distribution is plotted).

    Returns
    -------
    tuple (float, float) if output='simple'
        (posterior mean MIC, posterior std MIC) in µg/mL.
    ndarray if output='full'
        All valid (non-NaN) MIC posterior samples.
    """
    # Vectorise solve_mic so it can operate element-wise on arrays
    solve_mic_vec = np.vectorize(solve_mic)

    # Extract flattened posterior chains for each parameter
    r0_post    = idata_rep.posterior['r0'].values.flatten()
    q_post     = idata_rep.posterior['q'].values.flatten()
    c0_post    = idata_rep.posterior['c0'].values.flatten()
    alpha_post = idata_rep.posterior['alpha'].values.flatten()

    # Compute MIC for every posterior sample in parallel (vectorised)
    mic_post_full = solve_mic_vec(r0_post, q_post, c0_post, alpha_post)

    # Drop any samples where the solver failed to converge
    mic_clean = mic_post_full[~np.isnan(mic_post_full)]

    if output == 'full':
        return mic_clean
    else:
        MICavg = np.nanmean(mic_post_full)
        MICstd = np.nanstd(mic_post_full)
        return MICavg, MICstd
