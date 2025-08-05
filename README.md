# IR-Med - An Ad Hoc Information Retrieval Approach for Medicines’ Purchasing Public Notices

## Project Overview
IR-Med is an ad-hoc information retrieval system designed to support the auditing of public medicine procurement by matching medicine descriptions from procurement documents with official standardized records from the Brazilian CMED (Chamber of Drug Market Regulation) database.

This project was developed as part of a research effort to investigate reliable and scalable IR methods that can assist audit professionals when facing large volumes of data and limited resources. The solution focuses on identifying relevant drug items by preprocessing free-text medicine descriptions and comparing them to a clustered and normalized version of the CMED data.

The methodology is composed of two main phases:

### Modeling Phase
- **Text Preprocessing:** Cleans and normalizes the CMED data by:
    - Lowercasing and accent removal
    - Abbreviating forms using ANVISA's controlled vocabulary
    - Removing special characters, numbers (from active ingredients), stopwords, chemical ions, and duplicated terms
- **Clustering:** Groups CMED entries by active ingredient(s), storing them in a hash table for efficient lookup
- **Token Extraction:** Extracts and stores token sets for active ingredients and medicine presentation info (form, dosage, etc.)

### Information Retrieval Phase
- **Query Preprocessing:** Applies the same normalization steps to medicine descriptions found in procurement documents
- **Ingredient Matching:** Uses the Jaro-Winkler similarity metric to identify the closest matching cluster of active ingredients
- **Presentation Matching:** Computes the overlap between the presentation tokens and CMED entries to finalize matching results

## Versioning

This repository has been updated since the publication of the associated paper. These updates do not alter the core logic or methodology of the IR-MED solution. Instead, they focus on:

- Improving code readability and structure
- Refactoring to better follow the S.O.L.I.D. design principles
- Adding documentation (e.g., README files) to clarify the codebase, data handling, and data collection process

We recommend using the latest version of the repository for better maintainability and understanding.

If you need to access the exact version of the code used at the time of the paper's publication, it is available under the tag:

>v1.0-sbsi-2025

This version is also archived and citable via Zenodo:

[![DOI](https://zenodo.org/badge/851151662.svg)](https://doi.org/10.5281/zenodo.15850722).

## Repository Structure
```
ir-med/
├── data/                       # Example input files and lookup databases
├── README.md                   # This file
├── etl_functions.py            # ETL functions and data preparation
├── ir_med_example.ipynb        # Example notebook demonstrating the IR-Med pipeline
├── camelot_example.ipynb       # Example notebook showing how to extract tablem from PDFs using camelot
├── ir_med.py                   # Core IR matching logic
└── requirements.txt            # Dependencies
```

## Installation
You can set up the environment using pip and Python ≥ 3.8.

```
> git clone git@github.com:ArthurLimaS/ir-med.git
> cd ir-med
> python -m venv ir-med-venv
> source ir-med-venv/bin/activate  # or .\ir-med-venv\Scripts\activate on Windows
> pip install -r requirements.txt
```

## Citation
If you use this work, please cite the accompanying paper:

> Silva, A. L., Lima, A. M., Valença, G., & Cabral, G. G. (2025, May). Ad-hoc vs LLM based System for Information Retrieval in Large Tabular Data: A Comparative Study in Public Medicine Procurement Audits. In Simpósio Brasileiro de Sistemas de Informação (SBSI) (pp. 751-758). SBC.  
