# Build Plan: Intelligent Retail Customer Behavior Prediction and Segmentation System

Subject: 21CS521 Data Science and Analytics
Stack: Python + Streamlit, built using Google Antigravity
Dataset: UCI Online Retail (`Online Retail.xlsx`)

> Important: Every number shown in the app must be computed from your actual file by the profiling step. Do not copy figures from this document into your app, report, or slides.

---

## 0. Setup (before opening the agent, about 20 minutes)

1. Install **Python 3.10 or 3.11**. Python 3.12+ with NumPy 2 is where `scikit-learn-extra` and some older packages break.
2. Create the folder `retail_customer_intelligence/` and open it in Antigravity.
3. Download the UCI Online Retail dataset (UCI repository, dataset ID 352) and place it at `data/Online Retail.xlsx`. The agent may not be able to download it, so do this manually.
4. Save your full original project prompt as **`SPEC.md`** in the project root. It is the single source of truth, so you never have to re-paste it.
5. Create and activate a virtual environment:
   ```
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # Mac/Linux: source .venv/bin/activate
   ```

---

## 1. Antigravity configuration

### Rules vs workflows
- **Rules** are system-level instructions that guide the agent's behavior.
- **Workflows** are saved prompts you trigger on demand with `/` slash commands.
- Workspace locations:
  - Rules: `your-workspace/.agent/rules/`
  - Workflows: `your-workspace/.agent/workflows/`

(Menu names change between versions. If something differs in your build, trust what your version shows.)

### File 1: `.agent/rules/project-rules.md`
```
Read SPEC.md first and follow it.
Work on ONE phase at a time. Do not start the next phase until I confirm.
Never fabricate results. All metrics, charts, and statistics must be computed from the actual dataset.
No hard-coded dataset statistics anywhere.
Use random_state=42 for reproducibility.
Fit scalers and encoders on the training set only (no leakage).
Use Streamlit caching (st.cache_data / st.cache_resource) for heavy work.
Show clear, user-friendly error messages in the UI.
If something fails: identify the exact error, explain why, fix only the failing part, and give me the exact command to rerun. Do not rewrite unrelated files.
After each phase report: what was completed, files changed, how to run it, what I should understand, what remains.
Explain each important module in beginner-friendly language: what, why, input, internals, output, and what to say in viva.
```

### File 2: `.agent/workflows/next-phase.md`
```
Read SPEC.md and the project rules. Implement ONLY the phase I name.
Produce: files changed, how to run it, a test I can execute in the terminal, what remains.
Stop and wait for my confirmation before doing anything else.
```
Use it as: `/next-phase Phase 3: Classification`

### Modes
- **Planning mode:** the agent produces a task list and implementation plan before coding. Use it for Phases 1, 3, and 7 (design decisions and multi-file work).
- **Fast mode:** skips planning and executes immediately. Use it for small bug fixes and simple commands.

### Autonomy and safety
- Your spec requires the agent to pause after each phase, so avoid the "Always Proceed" setting (agent never asks for feedback). Use "Agent Decides" or a review-heavy setting.
- Keep a terminal **deny list** so dangerous commands are blocked even in Turbo/Auto mode.
- Review the **implementation plan artifact** before approving each phase.

### Verification
Use the browser sub-agent in Phase 7 to open `http://localhost:8501`, click through all pages, and capture screenshots. Treat these as evidence to review, not proof of correctness. Check the numbers yourself against the profiling output.

---

## 2. Technical decisions to lock in now

### 2.1 Cleaning strategy (split by use case)
Apriori works at invoice level, so rows with missing `CustomerID` are still useful there. Customer-level modules need `CustomerID`. Do **not** apply one blanket "drop nulls" step.

Transaction cleaning steps (log rows removed at each step):
1. Remove exact duplicate rows.
2. Remove cancelled invoices (`InvoiceNo` starting with `C`).
3. Remove `Quantity <= 0`.
4. Remove `UnitPrice <= 0`.
5. Remove non-product stock codes (for example POST, D, M, BANK CHARGES, DOT, CRUK; confirm against the actual data).
6. Remove missing product descriptions where needed (Apriori).
7. Create `Revenue = Quantity x UnitPrice`.

Customer-level data additionally requires non-null `CustomerID`.

### 2.2 Classification target (leakage-free, time-split design)
If the target is built from Monetary and Monetary or Quantity is also a feature, that is leakage. Use a **time split**:

- Choose a **cutoff date** (example: 2011-09-01; confirm against the actual date range).
- **Features**: computed ONLY from transactions before the cutoff.
- **Target**: customer's spend AFTER the cutoff (next ~3 months), converted into classes.
- Proposed classes (verify against actual distribution first):
  - **Inactive**: zero spend after cutoff
  - **Low/Medium**: below the median of active-customer future spend
  - **High**: above that median
  - Adjust so classes are not badly imbalanced. Report class counts.

**Customer feature table (observation window, before cutoff):**

| Feature | Meaning |
|---|---|
| Recency | Days from last purchase to cutoff |
| Frequency | Number of distinct invoices |
| Monetary | Total revenue |
| TotalQuantity | Sum of items bought |
| UniqueProducts | Distinct StockCodes bought |
| NumInvoices | Invoice count (check overlap with Frequency; drop one if identical) |
| AvgInvoiceValue | Monetary / invoices |
| AvgQtyPerInvoice | TotalQuantity / invoices |
| Country_UK | 1 if UK, else 0 |

**Leakage argument for viva:** features come strictly from the past, the target strictly from the future, the train/test split is done before scaling, and scalers are fit on training data only.

Expect moderate accuracy, not near 100%. That is a good sign. Compare against a **majority-class baseline** to show the models learn something.

### 2.3 Models
- **Decision Tree**: limit `max_depth` (and/or `min_samples_leaf`) to control overfitting. Show feature importance and a depth-limited tree plot.
- **Naive Bayes**: `GaussianNB` (continuous features).
- **SVM**: `Pipeline(StandardScaler, SVC(kernel="rbf"))`. Optionally compare with a linear kernel. Enable `probability=True` only if you explain it (Platt scaling, slower). Otherwise show the class only.
- Stratified train/test split. Report accuracy, precision, recall, F1 (state macro or weighted), and confusion matrices.
- Highlight best model by the metric the user selects.

### 2.4 Clustering
- Use log-transformed and scaled features.
- Choose K using elbow (inertia) plus silhouette over K = 2 to 8. Show both plots.
- **K-Means** from scikit-learn.
- **K-Medoids**: write your own simple PAM-style implementation using a precomputed distance matrix (about 40 lines). This is safer than `scikit-learn-extra`, which often fails to install. About 4,000 customers is small enough to run quickly. If you prefer a library, try `scikit-learn-extra` first and fall back to your own.
- Build a cluster profile table (mean of each feature per cluster). Interpret clusters from those statistics only. K-Means does not "know" what a high-value customer is.
- Compare K-Means vs K-Medoids (silhouette, cluster sizes, agreement).

### 2.5 Apriori
- Build an invoice x product boolean basket matrix.
- To avoid memory problems: restrict to top ~100 to 200 products and/or one country (UK).
- Start with `min_support` around 0.01 to 0.03, with UI sliders for support, confidence, and lift.
- Show frequent itemsets and rules with support, confidence, and lift.

### 2.6 PCA and SOM
- **PCA** on scaled features: original feature count, components kept, explained variance, cumulative variance, 2D scatter colored by cluster.
- **SOM (MiniSom)**: grid about 8x8 to 10x10 on scaled features. Show U-matrix (distance map), hit map, and cluster or class labels per cell. Wrap the import in try/except so a dependency problem does not crash the app. If MiniSom fails, a simple NumPy SOM is the fallback.
- SOM is unsupervised. PCA is not a prediction algorithm.

### 2.7 Speed and caching
- Reading the `.xlsx` is slow. Convert once to `data/online_retail.parquet` (or CSV) and reuse it.
- `st.cache_data` for data and feature tables; `st.cache_resource` for trained models.

### 2.8 Preprocessing demonstration (before vs after)
- **Missing values**: counts per column, handling per use case.
- **Duplicates**: count removed.
- **Invalid transactions**: cancelled, negative quantity, invalid price.
- **Outliers**: IQR detection, then percentile clipping (1st to 99th) on skewed features.
- **Transformation**: `log1p` on Monetary, Quantity, Frequency-type features.
- **Discretization**: `qcut` bins for Recency, Frequency, Monetary (for example Low/Medium/High), used for display and explanation.
- For each step, the UI states why it is used.

---

## 3. Project structure

```
retail_customer_intelligence/
├── app.py
├── requirements.txt
├── README.md
├── SPEC.md
├── PLAN.md
├── .agent/
│   ├── rules/project-rules.md
│   └── workflows/next-phase.md
├── data/
│   └── Online Retail.xlsx
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── classification.py
│   ├── clustering.py
│   ├── association.py
│   ├── pca_analysis.py
│   ├── som_analysis.py
│   └── visualization.py
├── models/
└── outputs/
```

`requirements.txt` (pin versions only if something breaks):
```
pandas
numpy
scikit-learn
matplotlib
seaborn
openpyxl
pyarrow
mlxtend
streamlit
minisom
```

---

## 4. Phase-by-phase plan with prompts

### Phase 1: Dataset profiling and ML problem design (about 1 hour)
**Prompt:**
```
Follow SPEC.md Phase 1. Write src/data_loader.py and a throwaway profile.py that loads the file
and prints: shape, columns, dtypes, missing values, duplicates, cancelled invoices, negative
quantities, invalid prices, unique customers, unique products, invoice count, and date range.
Then propose: the customer feature table, the classification target with a time-split cutoff
date, a leakage check, the exact ML problem statement, and the architecture.
Do NOT build the app yet. Wait for my confirmation.
```
**Acceptance:** Output matches the real file. You understand and agree with the target and cutoff.

### Phase 2: Preprocessing and feature engineering (2 to 3 hours)
Files: `preprocessing.py`, `feature_engineering.py`
**Prompt:**
```
Phase 2. Implement clean_transactions() returning the cleaned data and a log of rows removed per
step; outlier detection (IQR) and percentile clipping; log1p transformation; qcut discretization;
a before/after statistics structure for the UI; and build_customer_features(df, cutoff) that
returns the customer feature table and the future-spend target. Use only pre-cutoff data for
features. Cache the loaded data as parquet. Include a test script that prints the feature head,
class counts, and asserts no feature uses post-cutoff data.
```
**Acceptance:** Test script passes; class counts printed; no post-cutoff data in features.

### Phase 3: Classification (2 hours)
File: `classification.py`
**Prompt:**
```
Phase 3. Implement train_all_models() returning fitted Decision Tree, GaussianNB, and SVM
(StandardScaler + RBF) with accuracy, precision, recall, F1, confusion matrices, classification
reports, decision tree feature importances, and a depth-limited tree plot. Include a majority-class
baseline. Stratified split, random_state=42. Implement predict(model_name, feature_dict) that
applies the correct scaling. Show probabilities only for Decision Tree and Naive Bayes unless SVM
probability=True is deliberately enabled and explained. Print a comparison table in a test script.
```
**Acceptance:** Comparison table prints; manual prediction works for each model; baseline shown.

### Phase 4: Clustering (2 hours)
File: `clustering.py`
**Prompt:**
```
Phase 4. Implement elbow and silhouette scan (K=2..8), K-Means, and K-Medoids (own PAM-style
implementation with a precomputed distance matrix, or scikit-learn-extra if it installs cleanly).
Return labels, cluster sizes, silhouette scores, cluster profile table (feature means per cluster),
and rule-based interpretation labels derived ONLY from those statistics. Include a K-Means vs
K-Medoids comparison. Test script prints results.
```
**Acceptance:** Both algorithms run; sizes and silhouette displayed; profile table makes sense.

### Phase 5: Apriori (1 to 2 hours)
File: `association.py`
**Prompt:**
```
Phase 5. Implement Apriori on invoice-level data (InvoiceNo -> products). Apply cleaning first,
restrict to top N products and optionally one country to control memory. Return frequent
itemsets and association rules with support, confidence, and lift, with adjustable thresholds.
Do NOT run Apriori on the customer RFM table.
```
**Acceptance:** Runs in a few seconds; rules table is non-empty at sensible thresholds.

### Phase 6: PCA and SOM (2 hours)
Files: `pca_analysis.py`, `som_analysis.py`
**Prompt:**
```
Phase 6. PCA on scaled customer features: explained variance, cumulative variance, 2D projection
for plotting with optional cluster coloring. SOM with MiniSom on the same scaled features: train,
return U-matrix, hit map, and per-cell cluster/class distribution. Wrap SOM import in try/except
with a clear message on failure.
```
**Acceptance:** Variance table and 2D points available; SOM trains and plots render.

### Phase 7: Streamlit integration (4 to 5 hours)
Files: `app.py`, `visualization.py`
**Prompt:**
```
Phase 7. Build app.py with sidebar navigation and these pages: Dashboard, Dataset & Preprocessing,
Customer Behavior Prediction, Model Comparison, Customer Segmentation, Product Association
Analysis, PCA Visualization, SOM Visualization, About Project. Every module page starts with a
short WHAT / WHY / RESULT box. Use st.cache_data and st.cache_resource so models are not retrained
on page changes. Prediction page: input fields, option to select an existing customer, one button
per model, predicted class, model name, and probabilities only where valid. Show clear error
messages. App must run with: streamlit run app.py
```
**Acceptance:** All nine pages load with no errors. Use the browser sub-agent to click through and take screenshots.

### Phase 8: Testing (2 hours)
**Prompt:**
```
Phase 8. Test the whole app from a fresh virtual environment: pip install -r requirements.txt, then
streamlit run app.py. Check first run vs cached run, missing dataset file (clear message), unusual
input values on the prediction page, and that no statistic is hard-coded. Fix only what fails.
```
**Acceptance:** Clean-environment run works end to end.

### Phase 9: Presentation material (3 hours)
**Prompt:**
```
Phase 9. Produce: README.md (setup + dataset instructions), a 5 to 10 minute demo script following
the 13-step flow in SPEC.md, answers to the 27 viva questions, a screenshot checklist, a PPT
outline, and a project report outline. Use only results actually produced by the app.
```

---

## 5. Page requirements summary

| Page | Must show |
|---|---|
| Dashboard | Total transactions, customers, products, revenue, date range; revenue over time, top products, customers by country, purchase distribution |
| Dataset & Preprocessing | Profiling results; before vs after cleaning; outliers, transformation, discretization with reasons; customer feature table |
| Customer Behavior Prediction | Input fields or customer selector; three predict buttons; predicted class; model used; valid probabilities only |
| Model Comparison | Metrics table, bar chart, confusion matrices, best model highlighted by selected metric, baseline |
| Customer Segmentation | K-Means / K-Medoids selector; K selection plots; sizes; silhouette; PCA cluster plot; interpretation table |
| Product Association | Support/confidence/lift sliders; itemsets and rules tables |
| PCA | Original vs reduced features; explained and cumulative variance; 2D plot with optional cluster colors |
| SOM | Grid map, U-matrix, hit map, cluster distribution, simple explanation |
| About | Problem statement, target definition, leakage control, algorithm roles, limitations |

---

## 6. Timeline options

| Deadline | Approach |
|---|---|
| 3 days | Day 1: Phases 1 to 3. Day 2: Phases 4 to 7. Day 3: Phases 8 to 9. |
| 1 to 2 days | Merge Phases 4 to 6 into one prompt. Skip decision-tree plot, linear-SVM comparison, and Apriori sliders until the core works. |

**Never cut:** profiling, target explanation, three-model comparison, K-Means and K-Medoids, Apriori, PCA, SOM, prediction page.

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| `scikit-learn-extra` fails to install | Use own K-Medoids implementation |
| Apriori runs out of memory | Fewer products, higher min_support, or UK only |
| MiniSom install/import error | `pip install minisom`; otherwise simple NumPy SOM; app must not crash |
| Class imbalance inflates accuracy | Report per-class F1, use `class_weight="balanced"`, show baseline |
| Suspiciously high accuracy | Re-run the leakage check |
| Streamlit retrains models on every click | Confirm caching decorators on every heavy function |
| Agent rewrites unrelated files | Tell it: "Only modify the files related to this error" |
| Excel loading is slow | Use the parquet/CSV cache |

---

## 8. Final deliverables checklist

- [ ] Project runs with `streamlit run app.py` in a clean environment
- [ ] `requirements.txt` and `README.md` (with dataset download instructions)
- [ ] Profiling results visible in the UI
- [ ] Preprocessing before/after visible in the UI
- [ ] Model comparison table and chart, best model highlighted, baseline shown
- [ ] Predictions working for all three models
- [ ] K-Means and K-Medoids segmentation with interpretation table
- [ ] Apriori itemsets and rules
- [ ] PCA and SOM visualizations
- [ ] Architecture diagram (pipeline from SPEC.md)
- [ ] Screenshots of every page
- [ ] Viva Q&A document
- [ ] 5 to 10 minute demo script
- [ ] PPT and project report

---

## 9. Viva notes to remember

- Features come from before the cutoff and the target from after, so there is no leakage. Scalers are fit on training data only.
- Moderate accuracy is expected and honest. Compare against the majority-class baseline.
- Different algorithms solve different problems: classification predicts a label, clustering finds groups, Apriori finds co-purchased products, PCA reduces dimensions, SOM organizes similar customers on a map.
- Clusters are interpreted from feature statistics. K-Means does not know "high-value customer."
- PCA is not a prediction algorithm. SOM is unsupervised, not classification.
- Only quote numbers your own profiling step prints.
