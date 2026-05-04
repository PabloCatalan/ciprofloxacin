# A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions

This repository contains the data, models, and code necessary to reproduce the results and figures from the manuscript: 
**"A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions"** by Marina de la Fuente García, Javier Molina-Hernández, Pablo Chávez, Pablo Catalán, and Saúl Ares.

## Overview
Quantifying how antibiotics inhibit bacterial growth is central to the rational design of treatments and to tracking antimicrobial resistance. Here we derive a five-parameter logistic model for the action of ciprofloxacin, a widely used bactericidal antibiotic, by coarse-graining a chromosome-level mechanistic model of its interaction with DNA gyrase. 

Our model yields explicit analytical expressions for the drug-dependent growth rate and carrying capacity, capturing the characteristic nonlinear growth-inhibition curve of ciprofloxacin. It admits a closed-form solution that can be fitted directly to optical-density data across multiple drug concentrations simultaneously. 

Applying the model across a range of growth conditions—including varying carbon sources and bacteriostatic co-treatments—we find that the inferred parameters shift in biologically coherent ways. The minimum inhibitory concentration (MIC) emerges as an analytically derived quantity, and robust estimates can be obtained from as few as four concentrations. The model provides a unified, experimentally validated framework connecting chromosome-level drug mechanics to population-level dose-response.

## Repository Structure

* `functions.py`: Shared utilities for Bayesian inference, model simulation, and plotting used throughout the analysis. Contains the core logic for the five-parameter logistic model and MIC extraction.
* `cip_model_bayesian_inference.ipynb`: The core modeling notebook. It runs all MCMC fits and computes the intermediate results (NetCDF posteriors, CSV summary tables) across diverse conditions (glucose, glycerol, TMP, CHL).
* `main_figures.ipynb`: Generates the main figures of the manuscript (Figs. 1–4).
* `supp_figures.ipynb`: Generates all supplementary figures (Figs. S1–S6).
* `data/`: Directory containing the raw experimental OD data.
* `results/`: Directory where the intermediate posteriors (`.nc`) and parameter summaries (`.csv`) are saved by the inference notebook.
* `figures/`: Directory where the generated `.pdf` figures are saved.

## Installation

To ensure full reproducibility, we recommend using Conda to manage your Python environment. 

1. Clone the repository:
   git clone https://github.com/pablocatalan/cirpofloxacin.git
   cd ciprofloxacin

2. Create the Conda environment from the provided environment.yml file:
   conda env create -f environment.yml

3. Activate the environment:
   conda activate alternativa_cip

## Usage

**Important:** You must generate the Bayesian inference results before attempting to plot the figures. Please run the notebooks in the following order:

1. `cip_model_bayesian_inference.ipynb`
   * Run this notebook first.
   * It performs simultaneous model fits to the CIP concentrations, independent logistic fits, MIC robustness analysis, and fits across varying carbon sources and co-treatments.
   * Outputs will be automatically saved to the `results/` folder.

2. `main_figures.ipynb`
   * Reads pre-computed posteriors from `results/` and saves the main manuscript figures as PDFs in the `figures/` folder.

3. `supp_figures.ipynb`
   * Reads pre-computed posteriors and produces the supplementary figures, saving them to the `figures/` folder.

## Citation

If you use this code or model in your research, please cite our manuscript:

> de la Fuente García M, Molina-Hernández J, Chávez P, Catalán P, Ares S. (2024). A mechanistic logistic model of ciprofloxacin action captures growth-killing coupling across diverse conditions. [Journal Name]. DOI: [Insert DOI here]

## License
MIT License - see the LICENSE file for details.