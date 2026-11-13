# Retail Customer Intelligence

This project uses the UCI Online Retail transaction dataset to explore customer behaviour. It brings together customer classification, segmentation, market basket analysis, PCA, and Self-Organizing Maps in a Streamlit application.

The work is split into separate analysis modules so the data preparation and modelling steps can be inspected independently.

## What the project explores

- **Classification:** compare Decision Tree, Naive Bayes, and SVM models for predicting customer spend categories.
- **Customer segmentation:** apply K-Means and K-Medoids to group customers using RFM and behavioural features.
- **Market basket analysis:** use Apriori to find product associations, evaluated with support, confidence, and lift.
- **PCA:** reduce the feature space to two dimensions for visual inspection.
- **Self-Organizing Maps:** use a Kohonen map and U-Matrix to explore the structure of customer data.
- **Streamlit interface:** bring the analysis and visualisations together in an interactive application.

## Dataset

The source is the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), which contains transactions from a UK-based online retailer between December 2010 and December 2011.

The records include invoice, product, quantity, date, unit price, customer, and country fields. The project keeps the spreadsheet and a Parquet cache under `retail_customer_intelligence/data/`.

## How the time split works

The feature pipeline uses **2011-09-01** as a temporal cutoff:

- Transactions before the cutoff are used to calculate historical customer features, including Recency, Frequency, and Monetary value.
- Transactions after the cutoff are used to define the future spend target.
- Preprocessing steps such as scaling are fitted on the training data before being applied to held-out data.

This separation matters because using future transactions to calculate past customer features would leak information into the model. The detailed choices are documented in `SPEC.md`.

## Tools used

- Python, pandas, NumPy
- scikit-learn and scikit-learn-extra
- mlxtend (Apriori)
- MiniSom
- Streamlit
- Matplotlib and Seaborn
- PyArrow and OpenPyXL

## Project structure

```text
.
├── retail_customer_intelligence/
│   ├── data/
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── classification.py
│   │   ├── clustering.py
│   │   ├── association.py
│   │   ├── pca_analysis.py
│   │   ├── som_analysis.py
│   │   └── visualization.py
│   ├── app.py
│   ├── requirements.txt
│   ├── SPEC.md
│   ├── PRESENTATION_MATERIAL.md
│   └── verify_all.py
└── README.md
```

## Run locally

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/padmanaban-mu/retail-customer-intelligence.git
cd retail-customer-intelligence/retail_customer_intelligence
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the packages and start Streamlit:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal (usually [http://localhost:8501](http://localhost:8501)).

## Verification

Run the verification script from the application directory:

```bash
python verify_all.py
```

The repository also contains phase-specific tests for data preparation, feature engineering, classification, clustering, association analysis, PCA, and SOM.

## Interpreting the results

This dataset describes a particular retailer and a historical period; it is not a representation of current retail behaviour. Results depend on the selected time split, feature construction, preprocessing, and model settings. Re-run the verification scripts when changing the pipeline.

## License and dataset

Check the repository's license and the UCI dataset terms before redistributing the code or source data.
