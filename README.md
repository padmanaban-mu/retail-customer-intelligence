# Retail Customer Intelligence — Prediction & Segmentation

An end-to-end data science project that analyses online retail transactions to explore customer behaviour, predict future spend categories, segment customers, and discover product associations.

The project combines data preparation, supervised learning, unsupervised learning, association-rule mining, dimensionality reduction, and an interactive Streamlit application.

## Analytical modules

- **Dataset profiling and cleaning:** inspect transaction records, handle cancellations and missing customer identifiers, and prepare analysis-ready data.
- **Customer behaviour prediction:** compare Decision Tree, Naive Bayes, and Support Vector Machine models.
- **Model diagnostics:** evaluate classification performance with metrics and confusion matrices.
- **Customer segmentation:** use K-Means and K-Medoids clustering, with Elbow and Silhouette analysis to compare cluster configurations.
- **Market basket analysis:** apply Apriori to discover item associations using support, confidence, and lift.
- **Dimensionality reduction:** use PCA to project high-dimensional customer features.
- **Self-Organizing Maps:** use a Kohonen SOM and visualisations such as the U-Matrix to inspect topological structure.

## Dataset

The project uses the UCI Online Retail dataset (Dataset ID 352), containing transactions from a UK-based online retailer from December 2010 to December 2011.

The source data includes invoice, product, quantity, date, unit-price, customer, and country fields. The repository stores the source spreadsheet and a Parquet cache under `retail_customer_intelligence/data/`.

## Methodology and leakage prevention

The feature-engineering pipeline uses a temporal cutoff of **2011-09-01**:

- Transactions before the cutoff are used to compute customer features such as Recency, Frequency, Monetary value, and behavioural indicators.
- Transactions after the cutoff define the future spend target.
- Preprocessing bounds and scaling are fitted on training data and then applied to held-out data.

This time-based separation is important: using future transactions to construct historical features would leak target information into the model.

Heavy-tailed numeric features are clipped using empirical percentile bounds and transformed with `log1p` where appropriate. Consult `SPEC.md` for the detailed technical specification.

## Technology

- Python
- pandas and NumPy
- scikit-learn and scikit-learn-extra
- mlxtend
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

### Requirements

Python 3.10 or 3.11 is recommended. Git is also required.

```bash
git clone https://github.com/padmanaban-mu/retail-customer-intelligence.git
cd retail-customer-intelligence/retail_customer_intelligence
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies and launch the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (normally http://localhost:8501).

## Verification

Run the project's verification suite from the application directory:

```bash
python verify_all.py
```

Individual phase tests are also provided for preprocessing, feature engineering, classification, clustering, association analysis, PCA, and SOM.

## Reproducibility notes

The dataset is historical and reflects the source retailer and collection period; it should not be interpreted as current market behaviour. Model results depend on the selected time split, preprocessing, feature definitions, and random seeds. Re-run the verification scripts when changing the pipeline.

## License

No license file was identified during this documentation update. Check licensing and the dataset's terms before redistribution.
