import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def build_basket_matrix(df, top_n_products=150, country='United Kingdom'):
    """
    Transforms the transaction data into a boolean basket matrix (InvoiceNo x Products).
    Restricts to a specific country and top N products to prevent memory overflow.
    """
    # 1. Filter by country (optional but recommended for Apriori)
    if country:
        df_subset = df[df['Country'] == country].copy()
    else:
        df_subset = df.copy()
        
    # Drop rows without a valid Description
    df_subset = df_subset.dropna(subset=['Description'])
    
    # 2. Restrict to top N products by frequency
    # This prevents the matrix from having 4000+ columns, which crashes Apriori
    top_products = df_subset['Description'].value_counts().head(top_n_products).index
    df_subset = df_subset[df_subset['Description'].isin(top_products)]
    
    # 3. Build the basket (InvoiceNo x Description)
    # We sum the quantity just in case a product is listed twice in an invoice,
    # then we pivot it.
    basket = (df_subset
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))
              
    # 4. Convert quantities to booleans (1 if bought, 0 if not)
    # Apriori only cares IF an item was in the basket, not how many
    def encode_units(x):
        if x <= 0:
            return False
        if x > 0:
            return True
            
    # Apply the encoding across the dataframe efficiently
    basket_bool = basket.map(encode_units)
    
    return basket_bool

def run_apriori(basket_matrix, min_support=0.03, min_confidence=0.2, min_lift=1.0):
    """
    Runs the Apriori algorithm and generates association rules.
    Returns the frequent itemsets and the rules dataframe.
    """
    # 1. Generate Frequent Itemsets
    # use_colnames=True maps the column indices back to product names
    frequent_itemsets = apriori(basket_matrix, min_support=min_support, use_colnames=True)
    
    # Check if empty
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()
        
    # 2. Generate Association Rules
    # We initially filter by lift, then apply the confidence filter
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift, num_itemsets=len(frequent_itemsets))
    except TypeError:
        # Fallback for older versions of mlxtend
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    
    if not rules.empty:
        # 3. Filter by confidence
        rules = rules[rules['confidence'] >= min_confidence]
        
        # Sort by lift descending
        rules = rules.sort_values('lift', ascending=False)
        
    return frequent_itemsets, rules
