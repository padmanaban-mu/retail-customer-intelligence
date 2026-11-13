import pandas as pd
from src.data_loader import load_raw_data
from src.preprocessing import clean_transactions
from src.association import build_basket_matrix, run_apriori

def test_phase5():
    print("Loading and preparing data...")
    df_raw = load_raw_data()
    # Note: Apriori uses invoice-level data, so we don't drop rows missing CustomerID
    df_clean, _ = clean_transactions(df_raw)
    
    print("\nBuilding Basket Matrix (Top 100 UK Products)...")
    basket = build_basket_matrix(df_clean, top_n_products=100, country='United Kingdom')
    print(f"Basket Shape: {basket.shape} (Invoices x Products)")
    
    sparsity = (basket == False).sum().sum() / basket.size
    print(f"Matrix Sparsity: {sparsity:.2%}")
    
    print("\nRunning Apriori (Support=0.03, Confidence=0.2, Lift=1.0)...")
    itemsets, rules = run_apriori(basket, min_support=0.03, min_confidence=0.2, min_lift=1.0)
    
    print(f"\nFound {len(itemsets)} Frequent Itemsets.")
    if not itemsets.empty:
        print(itemsets.sort_values('support', ascending=False).head())
        
    print(f"\nFound {len(rules)} Association Rules.")
    if not rules.empty:
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        print("\nTop 5 Rules (Sorted by Lift):")
        top_rules = rules[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']].head()
        for _, row in top_rules.iterrows():
            print(f"IF [{row['antecedents_str']}] -> THEN [{row['consequents_str']}]")
            print(f"   Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")
            print("-" * 50)

if __name__ == "__main__":
    test_phase5()
