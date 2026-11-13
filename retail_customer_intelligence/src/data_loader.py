import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
EXCEL_PATH = os.path.join(DATA_DIR, 'Online Retail.xlsx')
PARQUET_PATH = os.path.join(DATA_DIR, 'online_retail.parquet')

def load_raw_data():
    if os.path.exists(PARQUET_PATH):
        print(f"Loading data from {PARQUET_PATH} (cached)...")
        return pd.read_parquet(PARQUET_PATH)
    elif os.path.exists(EXCEL_PATH):
        print(f"Loading data from {EXCEL_PATH} (this might take a while)...")
        df = pd.read_excel(EXCEL_PATH)
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        df['StockCode'] = df['StockCode'].astype(str)
        df['Description'] = df['Description'].astype(str)
        df['Country'] = df['Country'].astype(str)
        print(f"Saving to {PARQUET_PATH} for faster subsequent loads...")
        df.to_parquet(PARQUET_PATH)
        return df
    else:
        raise FileNotFoundError(f"Data file not found at {EXCEL_PATH}. Please download it.")
