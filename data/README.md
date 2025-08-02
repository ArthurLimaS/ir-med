# data/ Folder
This folder contains the datasets and inputs used in the IR-Med solution

## CMED

The **CMED** (Câmara de Regulação do Mercado de Medicamentos) table available at the time of the paper's publication. This table is published monthly by ANVISA and lists all medicines approved for sale in Brazil. Each row provides structured information such as the active ingredient, dosage, pharmaceutical form, barcode (EAN), manufacturer, among others.

>🔗 You can download the latest version of the CMED table from ANVISA's official website: https://www.gov.br/anvisa/pt-br/assuntos/medicamentos/cmed/precos

## Public Procurement Notices

This folder includes a few public procurement notices used to test the IR-Med system. Each notice is provided in two formats:

- The original PDF file
- A CSV file, which represents the cleaned and structured input used by the IR-Med pipeline

These notices were collected using the public API of the Tribunal de Contas do Estado de Pernambuco (TCE-PE).

> 🔗 API Endpoint: https://sistemas.tce.pe.gov.br/DadosAbertos/Exemplo!listar

> 📘 API Documentation (pt-BR): https://www.tcepe.tc.br/internet/index.php/dados-abertos

### Table Extraction Example
The Jupyter Notebook "camelot_example" is included in this folder to demonstrate how to use the camelot-py library to extract the tables from PDF files.