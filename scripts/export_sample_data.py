"""
Export sample data for Streamlit app forecasting.

This script creates a subset of the full featured data containing
representative store-item pairs for interactive forecasting in the
deployed Streamlit application.

Usage:
    python scripts/export_sample_data.py

Outputs:
    - data/processed/sample_forecast_data.pkl (~10-50 MB)
    - data/processed/store_item_lookup.csv

Author: Alberto Diaz Durana
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'processed'
    
    input_file = data_dir / 'full_featured_data.pkl'
    output_file = data_dir / 'sample_forecast_data.pkl'
    lookup_file = data_dir / 'store_item_lookup.csv'
    
    # Check input exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Run FULL_01_data_to_features.ipynb first.")
        sys.exit(1)
    
    print("=" * 60)
    print("EXPORT SAMPLE DATA FOR STREAMLIT APP")
    print("=" * 60)
    
    # Load full data
    print("\nStep 1: Loading full featured data...")
    df = pd.read_pickle(input_file)
    print(f"  Shape: {df.shape}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Select representative store-item pairs
    print("\nStep 2: Selecting representative store-item pairs...")
    print("  Strategy: 2 items per store (top seller + medium seller)")
    print("  Requirement: At least 100 days of history per item")
    
    sample_pairs = []
    
    for store in sorted(df['store_nbr'].unique()):
        store_data = df[df['store_nbr'] == store]
        
        # Get items with enough history and varied sales
        item_sales = store_data.groupby('item_nbr')['unit_sales'].agg(['mean', 'count'])
        item_sales = item_sales[item_sales['count'] >= 100]  # At least 100 days
        item_sales = item_sales.sort_values('mean', ascending=False)
        
        # Take top seller and one medium seller
        if len(item_sales) >= 2:
            top_item = item_sales.index[0]
            mid_idx = len(item_sales) // 2
            mid_item = item_sales.index[mid_idx]
            
            sample_pairs.append((store, top_item, 'top'))
            sample_pairs.append((store, mid_item, 'medium'))
            
            print(f"  Store {store}: items {top_item} (top), {mid_item} (medium)")
    
    print(f"\n  Total pairs selected: {len(sample_pairs)}")
    
    # Filter data to selected pairs
    print("\nStep 3: Filtering data to selected pairs...")
    mask = pd.Series(False, index=df.index)
    for store, item, _ in sample_pairs:
        mask |= ((df['store_nbr'] == store) & (df['item_nbr'] == item))
    
    df_sample = df[mask].copy()
    
    print(f"  Original rows: {len(df):,}")
    print(f"  Sample rows: {len(df_sample):,}")
    print(f"  Reduction: {(1 - len(df_sample)/len(df))*100:.1f}%")
    
    # Validate sample
    print("\nStep 4: Validating sample data...")
    print(f"  Stores: {df_sample['store_nbr'].nunique()}")
    print(f"  Items: {df_sample['item_nbr'].nunique()}")
    print(f"  Families: {df_sample['family'].nunique()}")
    print(f"  Date range: {df_sample['date'].min().date()} to {df_sample['date'].max().date()}")
    print(f"  Days: {df_sample['date'].nunique()}")
    
    families = df_sample['family'].unique().tolist()
    print(f"  Family list: {families}")
    
    # Save sample data
    print("\nStep 5: Saving sample data...")
    df_sample.to_pickle(output_file)
    file_size = output_file.stat().st_size / 1e6
    print(f"  Saved: {output_file}")
    print(f"  Size: {file_size:.1f} MB")
    
    # Warn if too large for Streamlit Cloud
    if file_size > 100:
        print(f"  WARNING: File size > 100 MB may cause issues on Streamlit Cloud")
    
    # Create lookup table for UI dropdowns
    print("\nStep 6: Creating store-item lookup table...")
    lookup = df_sample.groupby(['store_nbr', 'item_nbr', 'family']).agg({
        'unit_sales': ['mean', 'std', 'count']
    }).reset_index()
    lookup.columns = ['store_nbr', 'item_nbr', 'family', 'avg_sales', 'std_sales', 'n_days']
    lookup = lookup.round(2)
    lookup.to_csv(lookup_file, index=False)
    print(f"  Saved: {lookup_file}")
    print(f"  Pairs: {len(lookup)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  1. {output_file.name} ({file_size:.1f} MB)")
    print(f"  2. {lookup_file.name}")
    print(f"\nNext steps:")
    print(f"  1. Copy files to Streamlit app: ~/Demand-forecasting-in-retail-app/data/")
    print(f"  2. Update app to use sample data")
    print(f"  3. Test locally, then push and redeploy")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
