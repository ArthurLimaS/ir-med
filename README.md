# IR-Med - An Ad Hoc Information Retrieval Approach for Medicines’ Purchasing Public Notices

IR-MED is a proof-of-concept information retrieval system developed to support public auditing tasks, specifically the automated matching of drug specifications with registered pharmaceutical products. This repository contains the codebase for the ad-hoc solution described in the associated research.

# 🔍 Project Overview
Auditing public expenses often requires analyzing extensive documentation under resource constraints. This project proposes an automated system capable of matching non-standardized drug descriptions from public notices with structured records from a large pharmaceutical database (~25,000 rows).

This system implements an ad-hoc IR solution that:
- Leverages domain-specific preprocessing and similarity heuristics.
- Achieves accuracy between 72.4% and 86.9% depending on the configuration.

# 📚 Repository Structure
ir-med/
├── etl/                    # ETL functions and data preparation
├── ir_med/                 # Core IR matching logic
├── data/                   # Example input files and lookup databases
├── notebooks/              # Jupyter Notebooks for exploration and evaluation
├── tests/                  # Unit tests for core functions
├── requirements.txt        # Dependencies
├── README.md               # This file
└── main.py                 # Optional entry point (if available)

# ⚙️ Installation
You can set up the environment using pip and Python ≥ 3.8.

bash
git clone https://github.com/ArthurLimaS/ir-med.git
cd ir-med
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArthurLimaS/ir-med/blob/main/example_notebook.ipynb)

# Citation
If you use this work, please cite the accompanying paper.

> Silva, A. L., Lima, A. M., Valença, G., & Cabral, G. G. (2025, May). Ad-hoc vs LLM based System for Information Retrieval in Large Tabular Data: A Comparative Study in Public Medicine Procurement Audits. In Simpósio Brasileiro de Sistemas de Informação (SBSI) (pp. 751-758). SBC.  

The exact version used is tagged as `v1.0-sbsi-2025` and archived at [![DOI](https://zenodo.org/badge/851151662.svg)](https://doi.org/10.5281/zenodo.15850722).
