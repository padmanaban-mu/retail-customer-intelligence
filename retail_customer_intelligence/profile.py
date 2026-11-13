import pandas as pd
from src.data_loader import load_raw_data

def profile_data():
    try:
        df = load_raw_data()
    except FileNotFoundError as e:
        print(e)
        return

    print("----- Data Profiling Results -----")
    print(f"Shape: {df.shape}")
    print("\nColumns and Dtypes:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nExact Duplicate Rows: {df.duplicated().sum()}")
    
    # Identify cancelled invoices
    if 'InvoiceNo' in df.columns:
        cancelled = df['InvoiceNo'].astype(str).str.startswith('C').sum()
        print(f"\nCancelled Invoices (starting with 'C'): {cancelled}")
    
    # Negative/Zero quantities
    if 'Quantity' in df.columns:
        neg_zero_qty = (df['Quantity'] <= 0).sum()
        print(f"Negative or Zero Quantity: {neg_zero_qty}")
        
    # Invalid prices
    if 'UnitPrice' in df.columns:
        invalid_price = (df['UnitPrice'] <= 0).sum()
        print(f"Invalid Unit Price (<= 0): {invalid_price}")
        
    # Unique metrics
    if 'CustomerID' in df.columns:
        print(f"\nUnique Customers: {df['CustomerID'].nunique()}")
    if 'StockCode' in df.columns:
        print(f"Unique Products: {df['StockCode'].nunique()}")
    if 'InvoiceNo' in df.columns:
        print(f"Total Unique Invoices: {df['InvoiceNo'].nunique()}")
        
    # Date range
    if 'InvoiceDate' in df.columns:
        print(f"\nDate Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

if __name__ == "__main__":
    profile_data()
