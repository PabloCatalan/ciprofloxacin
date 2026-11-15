"""
functions.py
============
Shared code for the ciprofloxacin (CIP) dose-response analysis: model
equations, data loading, Bayesian inference, posterior summaries and the
plotting helpers used by more than one notebook.

All three notebooks start with ``from functions import *``:

  - cip_model_bayesian_inference.ipynb   all fits and intermediate results
  - main_figures.ipynb                   Figs. 1-5
  - supp_figures.ipynb                   Figs. S1-S7

Units: CIP concentrations are in ug/mL everywhere in the code and in the
saved results; figures display ng/mL (factor 1000).

Model (manuscript Section 2)
----------------------------
On the invariant manifold phi = phi*(c) the fork/cell system (Eq. 1) reduces
exactly to a logistic equation with (Eq. 13)

    h(c)   = 1 + (c/c0)**alpha
    Phi(c) = (sqrt(1 + psi*h) - 1) / (sqrt(1 + psi) - 1)
    r(c)   = r0 * Phi(c) * (1 - q*c*h) / h
    K(c)   = K0 * (1 - q*c*h)

Fixing the fork completion rate d = 1 1/h gives psi = 4*r0*(1 + r0) and the
five-parameter model (r0, K0, q, c0, alpha) used throughout the paper. The
MIC c* solves q*c*h(c*) = 1 (Eq. 15).
"""

import os

import arviz as az
import matplotlib
import matplotlib.gridspec as gridspec  # noqa: F401  (re-exported to the notebooks)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.integrate import solve_ivp  # noqa: F401  (re-exported to the notebooks)
from scipy.optimize import root_scalar

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# One seed for MCMC sampling and for the posterior draws used in plots, so
# that re-running the notebooks top to bottom gives identical output.
RANDOM_SEED = 89271458
rng = np.random.default_rng(RANDOM_SEED)

DATA_FILE = 'data/cip_dose_response_all.csv'
T_MAX = 12.0           # all fits use the first 12 h of each growth curve (h)
Y0_STANDARD = 0.001    # inoculum OD, first experimental series (standard medium)
Y0_MEDIA = 0.003       # inoculum OD, second series (glucose, glycerol, TMP, CHL)
MEDIA = ['glucose', 'glycerol', 'TMP', 'CHL']
MODEL_PARAMS = ['r0', 'K0', 'q', 'c0', 'alpha']
ALPHA_FIXED = 6.0      # centre of the narrow alpha prior in the fixed-alpha fits (Fig. S7)

# Wells excluded from every fit and every plot. On the CHL plate (day 251203)
# rows F and G failed to grow in all wells, including the drug-free
# controls: a plating failure, not a biological effect (see Methods).
EXCLUDED_ROWS = {'251203': ('F', 'G')}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(medium=None, t_max=T_MAX, path=DATA_FILE):
    """
    Load the smoothed dose-response data, keep t < t_max and drop the
    excluded wells (EXCLUDED_ROWS).

    Columns used downstream: 'medium', 'rep', 'day', 'well', 'dosis' (medium
    or co-treatment dose), 'dosisAb' (CIP, ug/mL), 'Minutes', 'Hours',
    'ODsmooth' (rolling-median OD600, see preprocessing_median.py).
    """
    df = pd.read_csv(path)
    keep = df['Hours'] < t_max
    if medium is not None:
        keep &= df['medium'] == medium
    day = df['day'].astype(str)
    row = df['well'].str[0]
    for bad_day, bad_rows in EXCLUDED_ROWS.items():
        keep &= ~((day == bad_day) & row.isin(bad_rows))
    return df[keep].copy()


def medium_conditions(df):
    """Medium/co-treatment doses present in df, highest first (fitting order)."""
    return sorted(df['dosis'].unique(), reverse=True)


# ---------------------------------------------------------------------------
# Model equations
# ---------------------------------------------------------------------------
_EPS = 1e-12


def hill(c, c0, alpha):
    """Hill factor h(c) = 1 + (c/c0)**alpha (Eq. 2)."""
    return 1.0 + (c / (c0 + _EPS)) ** alpha


def growth_params(c, r0, K0, q, c0, alpha, psi=None):
    """
    Concentration-dependent growth rate r(c) and carrying capacity K(c)
    (Eq. 13). With psi=None, d = 1 1/h is imposed so psi = 4*r0*(1 + r0).

    Above the MIC both r(c) and K(c) are negative: they share the factor
    1 - q*c*h(c). They are deliberately not clamped at zero (see g()).
    Works with numpy arrays and with PyMC tensors.
    """
    h = hill(c, c0, alpha)
    if psi is None:  # d = 1:  sqrt(1 + psi) = 1 + 2*r0
        Phi = (np.sqrt(1.0 + 4.0 * r0 * (1.0 + r0) * h) - 1.0) / (2.0 * r0 + _EPS)
    else:
        Phi = (np.sqrt(1.0 + psi * h) - 1.0) / (np.sqrt(1.0 + psi) - 1.0 + _EPS)
    surv = 1.0 - q * c * h
    return r0 * Phi * surv / h, K0 * surv


def logistic(t, y0, r, K):
    """Closed-form solution of dN/dt = r N (1 - N/K), N(0) = y0 (Eq. 14)."""
    return (K * y0) / (y0 + (K - y0) * np.exp(-r * t))


def g(c, t, y0, r0, K0, q, c0, alpha, psi=None):
    """
    Model prediction of OD600 at time t (h) and CIP concentration c (ug/mL):
    Eq. (14) with r(c), K(c) from Eq. (13).

    Above the MIC r and K are both negative, the denominator never crosses
    zero and N(t) decays monotonically to 0, so no clamping is needed.
    """
    r, K = growth_params(c, r0, K0, q, c0, alpha, psi)
    return logistic(t, y0, r, K)


def phi_star(c, r0, c0, alpha):
    """
    Fork density on the invariant manifold (forks per cell, Eq. 6) with
    d = 1 1/h, so that phi*(0) = r0:
        phi*(c) = (sqrt(1 + 4 r0 (1 + r0) h(c)) - 1) / 2
    """
    return 0.5 * (np.sqrt(1.0 + 4.0 * r0 * (1.0 + r0) * hill(c, c0, alpha)) - 1.0)


def two_ode_rhs(t, y, c, r0, K0, q, c0, alpha, d=1.0):
    """
    Right-hand side of the full fork/cell system (Eq. 1), y = [f, N],
    written in terms of the fitted parameters:
        lambda0 = r0 (1 + r0/d)   so that phi*(0) = r0/d,
        kappa   = q d.
    Used in Fig. S1 to check the logistic reduction.
    """
    f, N = y
    lam0 = r0 * (1.0 + r0 / d)
    kappa = q * d
    gN = 1.0 - N / K0
    dc = d / hill(c, c0, alpha)
    phi = f / N
    return [gN * (lam0 * N - dc * f) - kappa * c * phi * f,
            gN * dc * f - kappa * c * f]


# ---------------------------------------------------------------------------
# MIC
# ---------------------------------------------------------------------------

def solve_mic(q, c0, alpha):
    """
    MIC c* (ug/mL) for one parameter vector: the root of
        q c* (1 + (c*/c0)**alpha) = 1          (Eq. 15)
    found with Brent's method on [0, 1000] ug/mL. Returns np.nan if there is
    no root in that bracket.
    """
    def f(c):
        return q * c * (1.0 + (c / c0) ** alpha) - 1.0
    try:
        return root_scalar(f, bracket=[0.0, 1000.0], method='brentq').root
    except ValueError:
        return np.nan


_solve_mic_vec = np.vectorize(solve_mic)


def get_mic(idata, output='simple'):
    """
    Posterior distribution of the MIC, computed from every posterior sample.

    output='simple' -> (mean, sd) in ug/mL
    output='full'   -> array of all valid (non-NaN) MIC samples
    """
    post = idata.posterior
    mic = _solve_mic_vec(post['q'].values.ravel(),
                         post['c0'].values.ravel(),
                         post['alpha'].values.ravel())
    mic = mic[~np.isnan(mic)]
    if output == 'full':
        return mic
    return float(np.mean(mic)), float(np.std(mic))


# ---------------------------------------------------------------------------
# Bayesian inference
# ---------------------------------------------------------------------------

def _save(idata, filename):
    groups = [grp for grp in ('posterior', 'sample_stats') if grp in idata.groups()]
    idata.to_netcdf(filename, groups=groups)


def fit_model(DF, filename, y0, draws=1000, alpha_fixed=None, fit_psi=False,
              init='jitter+adapt_full', target_accept=0.95, refit=False):
    """
    Fit Eq. (14) simultaneously to all CIP curves in DF with one shared
    parameter set (PyMC, NUTS, 4 chains, 2000 tuning steps).

    If `filename` already exists and refit is False, the stored posterior is
    loaded instead of re-sampling. Delete the file (or pass refit=True) to
    fit again.

    Priors (Materials and Methods):
        r0    ~ HalfNormal(1)
        K0    ~ HalfNormal(1)
        q     ~ HalfNormal(10)
        c0    ~ LogNormal(log 0.01, 1)
        alpha ~ HalfNormal(4)           or Normal(alpha_fixed, 0.1) if given
        sigma ~ HalfNormal(0.1)         Normal likelihood on ODsmooth
    fit_psi=True adds psi ~ HalfNormal(4) (six-parameter model, Section 2;
    not used for any figure).

    DF needs columns 'dosisAb' (ug/mL), 'Hours' and 'ODsmooth'.
    Returns an arviz.InferenceData.
    """
    if os.path.exists(filename) and not refit:
        return az.from_netcdf(filename)

    with pm.Model():
        c = pm.Data('c', DF['dosisAb'].values)
        t = pm.Data('t', DF['Hours'].values)

        r0 = pm.HalfNormal('r0', sigma=1.0)
        K0 = pm.HalfNormal('K0', sigma=1.0)
        q = pm.HalfNormal('q', sigma=10.0)
        c0 = pm.LogNormal('c0', mu=np.log(0.01), sigma=1.0)
        if alpha_fixed is None:
            alpha = pm.HalfNormal('alpha', sigma=4.0)
        else:
            alpha = pm.Normal('alpha', mu=alpha_fixed, sigma=0.1)
        psi = pm.HalfNormal('psi', sigma=4.0) if fit_psi else None
        sigma = pm.HalfNormal('sigma', sigma=0.1)

        pm.Normal('OD', mu=g(c, t, y0, r0, K0, q, c0, alpha, psi),
                  sigma=sigma, observed=DF['ODsmooth'].values)

        idata = pm.sample(draws=draws, tune=2000, target_accept=target_accept,
                          cores=4, init=init, random_seed=RANDOM_SEED)
    _save(idata, filename)
    return idata


def fit_logistic(DF, filename, y0, refit=False):
    """
    Plain logistic fit to a single growth curve (one CIP concentration), the
    comparison model of Fig. 1B. Priors r ~ U(0, 2), K ~ U(0, 1).
    Loads `filename` instead if it exists and refit is False.
    """
    if os.path.exists(filename) and not refit:
        return az.from_netcdf(filename)

    with pm.Model():
        t = pm.Data('t', DF['Hours'].values)
        K = pm.Uniform('K', lower=0.0, upper=1.0)
        r = pm.Uniform('r', lower=0.0, upper=2.0)
        sigma = pm.HalfNormal('sigma')
        pm.Normal('OD', mu=logistic(t, y0, r, K), sigma=sigma,
                  observed=DF['ODsmooth'].values)
        idata = pm.sample(draws=1000, tune=2000, target_accept=0.95, cores=4,
                          random_seed=RANDOM_SEED)
    _save(idata, filename)
    return idata


def fit_linear(x, y, filename, target_accept=0.95, refit=False):
    """
    Bayesian linear regression y = a + b x (used for q vs r0 in Figs. 5A
    and S7A). Priors a, b ~ Normal(0, 40), sigma ~ HalfNormal(1).
    """
    if os.path.exists(filename) and not refit:
        return az.from_netcdf(filename)

    with pm.Model():
        a = pm.Normal('a', sigma=40)
        b = pm.Normal('b', sigma=40)
        sigma = pm.HalfNormal('sigma')
        pm.Normal('y_obs', mu=a + b * np.asarray(x), sigma=sigma, observed=np.asarray(y))
        idata = pm.sample(draws=1000, tune=2000, target_accept=target_accept,
                          cores=4, random_seed=RANDOM_SEED)
    _save(idata, filename)
    return idata


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------

def diagnostics(idata):
    """Max R-hat, min bulk ESS and number of divergences of a fit."""
    names = [v for v in MODEL_PARAMS + ['psi', 'sigma'] if v in idata.posterior]
    rhat = az.rhat(idata, var_names=names)
    ess = az.ess(idata, var_names=names)
    return {'max_rhat': max(float(rhat[v]) for v in names),
            'min_ess': min(float(ess[v]) for v in names),
            'n_divergent': int(idata.sample_stats['diverging'].values.sum())}


def posterior_draws(idata, n=1000, var_names=None):
    """n random posterior samples (with replacement) as a dict of arrays."""
    post = idata.posterior.stack(sample=('chain', 'draw'))
    var_names = var_names or [v for v in MODEL_PARAMS + ['psi'] if v in post]
    idx = rng.integers(0, post.sizes['sample'], n)
    return {v: post[v].values[idx] for v in var_names}


def summarise_posterior(idata, params=MODEL_PARAMS + ['MIC']):
    """
    Posterior median and 2.5% / 97.5% quantiles of each parameter and of the
    MIC (Table 1). Returns {'r0_median': ..., 'r0_lower': ..., ...}.
    """
    out = {}
    for par in params:
        x = get_mic(idata, 'full') if par == 'MIC' else idata.posterior[par].values.ravel()
        lower, median, upper = np.quantile(x, [0.025, 0.5, 0.975])
        out.update({f'{par}_median': median, f'{par}_lower': lower, f'{par}_upper': upper})
    return out


def rK_summary(idata, doses, n=1000):
    """
    Posterior mean and SD of r(c) and K(c) at each dose (Fig. 1C-D). Samples
    above the MIC (K < 0) are set to r = K = 0 so the summaries are physical.
    Returns arrays Ravg, Rstd, Kavg, Kstd.
    """
    p = posterior_draws(idata, n)
    Ravg, Rstd, Kavg, Kstd = [], [], [], []
    for c in doses:
        r, K = growth_params(c, p['r0'], p['K0'], p['q'], p['c0'], p['alpha'], p.get('psi'))
        dead = K < 0
        r, K = np.where(dead, 0.0, r), np.where(dead, 0.0, K)
        Ravg.append(r.mean()); Rstd.append(r.std())
        Kavg.append(K.mean()); Kstd.append(K.std())
    return map(np.array, (Ravg, Rstd, Kavg, Kstd))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def dose_colors(doses):
    """One 'inferno' colour per CIP dose, in increasing dose order."""
    doses = sorted(doses)
    cmap = plt.get_cmap('inferno', len(doses))
    return {dose: cmap(i) for i, dose in enumerate(doses)}


def plot_data(DF, ax):
    """
    Median OD600 across wells/replicates (line) and interquartile range
    (band) for each CIP dose. Returns {label: handle} for ax.legend().
    Legend labels are in ng/mL.
    """
    colors = dose_colors(DF['dosisAb'].unique())
    stats = (DF.groupby(['dosisAb', 'Minutes'])['ODsmooth']
               .agg(OD_median='median',
                    OD_q1=lambda x: x.quantile(0.25),
                    OD_q3=lambda x: x.quantile(0.75))
               .reset_index())
    stats['Hours'] = stats['Minutes'] / 60

    for dose, sub in stats.groupby('dosisAb'):
        ax.plot(sub['Hours'], sub['OD_median'], label=1000 * dose, color=colors[dose])
        ax.fill_between(sub['Hours'], sub['OD_q1'], sub['OD_q3'],
                        color=colors[dose], alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    ax.set_xlabel('time (hours)')
    ax.set_ylabel('OD$_{600}$')
    return dict(zip(labels, handles))


def _plot_band(ax, T, sims, color):
    mean, sd = sims.mean(axis=0), sims.std(axis=0)
    ax.plot(T, mean, ls='--', color=color)
    ax.fill_between(T, mean - sd, mean + sd, color=color, alpha=0.25)


def plot_pred(ax, filename, doses, y0, n=1000):
    """
    Posterior-predictive model curves: for each dose, the mean +/- 1 SD of
    g() over n random posterior samples (dashed line and band).
    """
    T = np.linspace(0, T_MAX, 100)
    colors = dose_colors(doses)
    p = posterior_draws(az.from_netcdf(filename), n)
    col = {k: v[:, None] for k, v in p.items()}
    for c in sorted(doses):
        sims = g(c, T[None, :], y0, col['r0'], col['K0'], col['q'], col['c0'],
                 col['alpha'], col.get('psi'))
        _plot_band(ax, T, sims, colors[c])


def logistic_filename(rep, c):
    return f'results/cip_logistic_standard_{rep}_dose_{c}_12h.nc'


def plot_pred_logistic(rep, ax, doses, y0, n=1000):
    """As plot_pred() for the independent per-dose logistic fits (Fig. 1B)."""
    T = np.linspace(0, T_MAX, 100)
    colors = dose_colors(doses)
    for c in sorted(doses):
        p = posterior_draws(az.from_netcdf(logistic_filename(rep, c)), n, ['r', 'K'])
        sims = logistic(T[None, :], y0, p['r'][:, None], p['K'][:, None])
        _plot_band(ax, T, sims, colors[c])


def _mark_mic(ax, mic, mic_sd, ymin, ymax, text_x, text_y):
    ax.vlines(mic, ymin=ymin, ymax=ymax, ls='dashed', color='k', alpha=0.5)
    ax.fill_between([mic - mic_sd, mic + mic_sd], ymin, ymax, alpha=0.1, color='k')
    ax.text(text_x, text_y, 'MIC (predicted)', rotation=90, fontsize=15)


def replicate_figure(rep, mic_text_dx=0.001):
    """
    Four-panel figure for one replicate of the standard condition
    (Fig. 1 for rep 2; Figs. S2-S5 for reps 1, 3, 4, 5):
      A  data + simultaneous model fit
      B  data + independent logistic fits
      C  r(c), model vs logistic
      D  K(c), model vs logistic
    Reads results/cip_model_standard_{rep}_12h.nc, the logistic posteriors
    and results/rK_{rep}_12h.csv.
    """
    DF = load_data('standard')
    DF = DF[DF['rep'] == rep]
    doses = sorted(DF['dosisAb'].unique())
    filename = f'results/cip_model_standard_{rep}_12h.nc'

    fig = plt.figure(figsize=(12, 8))

    for k, (label, predict) in enumerate([
            ('model', lambda ax: plot_pred(ax, filename, doses, Y0_STANDARD)),
            ('logistic', lambda ax: plot_pred_logistic(rep, ax, doses, Y0_STANDARD))]):
        ax = fig.add_subplot(2, 2, k + 1)
        by_label = plot_data(DF, ax)
        predict(ax)
        if label == 'logistic':
            ax.legend(by_label.values(), by_label.keys(),
                      title='CIP dose (ng/mL)', ncols=1, loc='upper left')
        ax.set_xlim([0, 12])
        ax.set_ylim([-0.01, 0.5])
        ax.text(4, 0.45, label, fontsize=15)
        ax.tick_params(labelsize=15)
        ax.text(0.0, 1.05, 'AB'[k], fontsize=20, transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), fontsize=15)
        ax.set_ylabel(ax.get_ylabel(), fontsize=15)

    mic, mic_sd = get_mic(az.from_netcdf(filename))
    rK = pd.read_csv(f'results/rK_{rep}_12h.csv')
    colors = plt.get_cmap('tab20')
    ticks = np.arange(0, 0.031, 0.005)

    panels = [('C', 'R', 'growth rate', (-0.05, 1.9), 0.5, 0),
              ('D', 'K', 'carrying capacity', (-0.03, 0.8), 0.25, 2)]
    for k, (label, var, ylabel, ylim, text_y, ci) in enumerate(panels):
        ax = fig.add_subplot(2, 2, k + 3)
        ax.errorbar(rK.c, rK[var + 'avg'], yerr=rK[var + 'std'], marker='o',
                    color=colors(ci), label='model')
        ax.errorbar(rK.c, rK[var + 'avgL'], yerr=rK[var + 'stdL'], marker='o',
                    color=colors(ci + 1), label='logistic')
        ax.set_ylim(ylim)
        _mark_mic(ax, mic, mic_sd, ylim[0], ylim[1], mic + mic_text_dx, text_y)
        ax.set_xlabel('dose (ng/mL)', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.legend(fontsize=15)
        ax.tick_params(labelsize=15)
        ax.text(0.0, 1.05, label, fontsize=20, transform=ax.transAxes)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{1000 * z:.1f}' for z in ticks])

    fig.tight_layout()
    return fig


MEDIUM_COLORS = {'glucose': 'blue', 'glycerol': 'green', 'TMP': 'red', 'CHL': 'purple'}


def plot_linear_band(ax, idata, xp, color, n=1000):
    """Mean +/- 1 SD of a + b x over n posterior samples of fit_linear()."""
    p = posterior_draws(idata, n, ['a', 'b'])
    sims = p['a'][:, None] + p['b'][:, None] * xp[None, :]
    mean, sd = sims.mean(axis=0), sims.std(axis=0)
    ax.plot(xp, mean, ls='-', color=color)
    ax.fill_between(xp, mean - sd, mean + sd, color=color, alpha=0.25)


def plot_q_and_mic_vs_r0(ax_q, ax_mic, table, linear_fits, fontsize=20):
    """
    q vs r0 (with the linear trends in `linear_fits`, {medium: idata}) and
    MIC vs r0 for every medium condition, with 95% credible intervals.
    `table` is a parameters table from summarise_posterior().
    Shared by Fig. 5A/C and Fig. S7.
    """
    def err(par, df):
        return [df[f'{par}_median'] - df[f'{par}_lower'],
                df[f'{par}_upper'] - df[f'{par}_median']]

    for medium in MEDIA:
        df = table[table['medium'] == medium]
        kw = dict(fmt='o', label=medium, color=MEDIUM_COLORS[medium], alpha=0.7)
        ax_q.errorbar(df['r0_median'], df['q_median'], xerr=err('r0', df),
                      yerr=err('q', df), **kw)
        ax_mic.errorbar(df['r0_median'], df['MIC_median'], xerr=err('r0', df),
                        yerr=err('MIC', df), **kw)

    xp = np.linspace(0.5, 1.2, 100)
    for medium, idata in linear_fits.items():
        plot_linear_band(ax_q, idata, xp, MEDIUM_COLORS[medium])

    ax_q.set_xlabel(r'$r_0$', fontsize=fontsize)
    ax_q.set_ylabel(r'$q$', fontsize=fontsize)
    ax_q.tick_params(labelsize=fontsize)

    ax_mic.set_xlabel(r'$r_0$', fontsize=fontsize)
    ax_mic.set_ylabel('MIC (ng/mL)', fontsize=fontsize)
    ax_mic.set_yscale('log')
    ax_mic.set_yticks([0.01, 0.1, 1])
    ax_mic.set_yticklabels([10, 100, 1000])
    ax_mic.tick_params(labelsize=fontsize)
