import pandas as pd
import numpy as np

def clean_transactions(df):
    """
    Cleans the transaction data based on specific rules and logs the impact.
    Returns the cleaned DataFrame and a dictionary detailing rows removed.
    """
    initial_len = len(df)
    log = {}
    
    # 1. Remove exact duplicates
    df_cleaned = df.drop_duplicates()
    log['duplicates_removed'] = initial_len - len(df_cleaned)
    current_len = len(df_cleaned)
    
    # 2. Remove cancelled invoices (InvoiceNo starts with 'C')
    # InvoiceNo should be string based on data_loader
    df_cleaned = df_cleaned[~df_cleaned['InvoiceNo'].astype(str).str.startswith('C')]
    log['cancelled_removed'] = current_len - len(df_cleaned)
    current_len = len(df_cleaned)
    
    # 3. Remove Quantity <= 0
    df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]
    log['negative_qty_removed'] = current_len - len(df_cleaned)
    current_len = len(df_cleaned)
    
    # 4. Remove UnitPrice <= 0
    df_cleaned = df_cleaned[df_cleaned['UnitPrice'] > 0]
    log['invalid_price_removed'] = current_len - len(df_cleaned)
    current_len = len(df_cleaned)
    
    # 5. Remove non-product stock codes (e.g., POST, D, M, DOT, CRUK)
    # Using a list of known non-product codes from UCI dataset
    non_product_codes = ['POST', 'D', 'M', 'PADS', 'DOT', 'CRUK', 'BANK CHARGES']
    df_cleaned = df_cleaned[~df_cleaned['StockCode'].isin(non_product_codes)]
    log['non_product_codes_removed'] = current_len - len(df_cleaned)
    current_len = len(df_cleaned)
    
    # 6. Create Revenue column
    df_cleaned['Revenue'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']
    
    return df_cleaned, log

def clean_for_apriori(df):
    """
    Cleans transactions for Market Basket Analysis.
    Keeps rows with missing CustomerID because we only care about what was in the invoice.
    Removes cancelled orders (C), removes negatives, removes non-products.
    """
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['Description'])
    df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str)
    df_clean = df_clean[~df_clean['InvoiceNo'].str.startswith('C')]
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    non_product_codes = ['POST', 'D', 'M', 'PADS', 'DOT', 'CRUK', 'BANK CHARGES']
    df_clean = df_clean[~df_clean['StockCode'].isin(non_product_codes)]
    return df_clean

def fit_outlier_bounds(df, columns):
    """
    Fits and returns the 1st and 99th percentile bounds for the given columns.
    """
    bounds = {}
    for col in columns:
        if col in df.columns:
            bounds[col] = {
                'lower': df[col].quantile(0.01),
                'upper': df[col].quantile(0.99)
            }
    return bounds

def apply_outlier_bounds(df, bounds):
    """
    Clips the dataframe columns based on pre-fitted bounds.
    """
    df_clipped = df.copy()
    for col, b in bounds.items():
        if col in df_clipped.columns:
            df_clipped[col] = df_clipped[col].clip(lower=b['lower'], upper=b['upper'])
    return df_clipped

def handle_outliers(df, columns):
    """
    Detects outliers using IQR (percentiles) and clips them between the 1st and 99th percentiles.
    Returns the dataframe with clipped columns.
    """
    bounds = fit_outlier_bounds(df, columns)
    return apply_outlier_bounds(df, bounds)

def apply_log1p_transform(df, columns):
    """
    Applies log1p transformation to handle highly skewed right-tailed data.
    """
    df_transformed = df.copy()
    for col in columns:
        if col in df_transformed.columns:
            # Ensure no negative values before log1p
            df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
    return df_transformed

def apply_qcut_discretization(df, column, q=3, labels=['Low', 'Medium', 'High']):
    """
    Bins continuous data into quantiles for display and explanation purposes.
    """
    df_binned = df.copy()
    if column in df_binned.columns:
        # drop_duplicates=True in case quantiles overlap
        df_binned[f'{column}_Binned'] = pd.qcut(df_binned[column], q=q, labels=labels, duplicates='drop')
    return df_binned

def get_before_after_stats(df_before, df_after):
    """
    Generates summary statistics for the UI to demonstrate the effect of cleaning.
    """
    return {
        'rows_before': len(df_before),
        'rows_after': len(df_after),
        'customers_before': df_before['CustomerID'].nunique() if 'CustomerID' in df_before.columns else 0,
        'customers_after': df_after['CustomerID'].nunique() if 'CustomerID' in df_after.columns else 0,
        'revenue_before': (df_before['Quantity'] * df_before['UnitPrice']).sum() if 'Quantity' in df_before.columns else 0,
        'revenue_after': df_after['Revenue'].sum() if 'Revenue' in df_after.columns else 0
    }
