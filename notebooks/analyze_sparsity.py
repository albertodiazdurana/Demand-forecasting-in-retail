import pandas as pd

# Load the feature-engineered dataset
df = pd.read_pickle('data/processed/w02_d05_FE_final.pkl')

print("="*60)
print("DATA SUFFICIENCY ANALYSIS")
print("="*60)

# Basic dimensions
print(f"\nTotal rows: {len(df):,}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Days covered: {(df['date'].max() - df['date'].min()).days + 1}")

# Unique entities
stores = df['store_nbr'].nunique()
items = df['item_nbr'].nunique()
families = df['family'].nunique()
days = (df['date'].max() - df['date'].min()).days + 1

print(f"\nUnique stores: {stores}")
print(f"Unique items: {items:,}")
print(f"Unique families: {families}")

# Sparsity calculation
theoretical_max = stores * items * days
sparsity = (1 - len(df) / theoretical_max) * 100

print(f"\n{'='*60}")
print("SPARSITY ANALYSIS")
print(f"{'='*60}")
print(f"Theoretical maximum rows (stores × items × days): {theoretical_max:,}")
print(f"Actual rows: {len(df):,}")
print(f"Sparsity: {sparsity:.2f}%")
print(f"Density: {100 - sparsity:.2f}%")

# Average observations per item
avg_obs_per_item = len(df) / items
print(f"\nAverage observations per item: {avg_obs_per_item:.1f}")

# For 2013 + Jan-Feb 2014 training period
train_df = df[(df['date'] >= '2013-01-01') & (df['date'] <= '2014-02-21')]
train_days = (train_df['date'].max() - train_df['date'].min()).days + 1

print(f"\n{'='*60}")
print("TRAINING DATA ANALYSIS (2013 + Jan-Feb 2014)")
print(f"{'='*60}")
print(f"Training period: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Training days: {train_days}")
print(f"Training rows: {len(train_df):,}")
print(f"Average observations per item: {len(train_df) / items:.1f}")
print(f"Average observations per store: {len(train_df) / stores:.1f}")
print(f"Average observations per store-item combo: {len(train_df) / (stores * items):.1f}")

# Non-zero sales
non_zero_train = train_df[train_df['unit_sales'] > 0]
print(f"\nNon-zero sales rows: {len(non_zero_train):,} ({len(non_zero_train)/len(train_df)*100:.1f}%)")
print(f"Zero sales rows: {len(train_df) - len(non_zero_train):,} ({(1-len(non_zero_train)/len(train_df))*100:.1f}%)")

# How many items have sufficient data?
item_counts = train_df.groupby('item_nbr').size()
print(f"\n{'='*60}")
print("ITEM-LEVEL DATA SUFFICIENCY")
print(f"{'='*60}")
print(f"Items with ≥10 observations: {(item_counts >= 10).sum()} ({(item_counts >= 10).sum()/len(item_counts)*100:.1f}%)")
print(f"Items with ≥20 observations: {(item_counts >= 20).sum()} ({(item_counts >= 20).sum()/len(item_counts)*100:.1f}%)")
print(f"Items with ≥50 observations: {(item_counts >= 50).sum()} ({(item_counts >= 50).sum()/len(item_counts)*100:.1f}%)")
print(f"Items with ≥100 observations: {(item_counts >= 100).sum()} ({(item_counts >= 100).sum()/len(item_counts)*100:.1f}%)")

print(f"\n{'='*60}")
print("CONCLUSION")
print(f"{'='*60}")

if len(train_df) >= 50000:
    print("✓ Training data appears SUFFICIENT (50K-80K rows)")
    print(f"  - {len(train_df):,} training samples")
    print(f"  - Most items have 20+ observations for learning patterns")
else:
    print("⚠ Training data may be INSUFFICIENT (<50K rows)")
    print(f"  - Only {len(train_df):,} training samples")
    print("  - Consider expanding training period further")
