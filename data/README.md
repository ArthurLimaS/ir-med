# ```data/``` Folder
This folder contains the datasets and input files used in the IR-Med solution.

## ```cmed/``` subfolder

The **CMED** (Câmara de Regulação do Mercado de Medicamentos) is a table published monthly by ANVISA, listing all medicines approved for sale in Brazil. Each row provides structured information such as the active ingredient, dosage, pharmaceutical form, barcode (EAN), manufacturer, among others.

Although ANVISA provides these files in spreadsheet format, they include warning lines before the actual data begins, which complicates automated processing. To address this, we manually create a cleaned version of the file:

- Removed non-tabular header/warning rows
- Retained only the columns relevant for medicine identification

The subfolder contains the following:

- **xls_conformidade_gov_2024_03.xls:** Original file downloaded from ANVISA
- **cmed_clean_2024_03.xls:** Cleaned version with standardized structure and reduced columns
- **cmed_clean_2024_03.csv:** CSV export of the cleaned file (separator = ';')

### CMED Version

The CMED data provided corresponds to the version available at the time of the paper’s publication. To access the latest CMED files, visit the official ANVISA portal:

>https://www.gov.br/anvisa/pt-br/assuntos/medicamentos/cmed/precos

## ```notices/``` subfolder

This folder also includes a few public procurement notices used to test the IR-Med pipeline. Each notice included inside ```/notices``` is provided in two formats:

- PDF: Original document
- CSV: Structured and cleaned version used as input for the IR-Med pileline

### TCE-PE "Dados Abertos" API

These documents were collected via the public API from the Tribunal de Contas do Estado de Pernambuco (TCE-PE):

> **API Endpoint:** https://sistemas.tce.pe.gov.br/DadosAbertos/Exemplo!listar

> **API Documentation (pt-BR):** https://www.tcepe.tc.br/internet/index.php/dados-abertos

### Table Extraction Example
To demonstrate how tables were extracted from the PDF notices, a sample Jupyter Notebook named camelot_example.ipynb is provided. It shows how to use the camelot-py library for PDF table extraction.