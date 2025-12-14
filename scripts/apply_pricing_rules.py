import pandas as pd
import numpy as np
import os

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'dataSet', 'revenue_lift_9pct_30000.csv')
OUTPUT_CSV = os.path.join(ROOT, 'revenue_comparison.csv')

# Load
df = pd.read_csv(DATA_PATH)

# Basic date handling if present
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DAY'] = df['Date'].dt.day
else:
    df['DAY'] = 1

# Create rule_price
df['rule_price'] = df['Price'].copy()
# Time-based rules
# Weekend indicator may not exist; default to 0
if 'is_weekend' not in df.columns:
    df['day_of_week'] = pd.to_datetime(df['Date'], errors='coerce').dt.dayofweek.fillna(0).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# Apply time-based multipliers
df.loc[df['is_weekend'] == 1, 'rule_price'] *= 1.05
if 'Season' in df.columns:
    df.loc[df['Season'] == 'Festival', 'rule_price'] *= 1.10
# Month-end
df.loc[df['DAY'] >= 25, 'rule_price'] *= 1.02
# Low demand discount
if 'Conversion Rate %' in df.columns:
    df.loc[df['Conversion Rate %'] < 1, 'rule_price'] *= 0.95

# Inventory-based rules
if 'Stock Level' in df.columns:
    df.loc[df['Stock Level'] < 20, 'rule_price'] *= 1.07
    df.loc[(df['Stock Level'] >= 20) & (df['Stock Level'] <= 60), 'rule_price'] *= 1.03
    df.loc[df['Stock Level'] > 200, 'rule_price'] *= 0.90
    # inventory_ratio
    df['inventory_ratio'] = df['Stock Level'] / (df['Units Sold'] + 1e-6)
    df.loc[df['inventory_ratio'] < 0.6, 'rule_price'] *= 1.05
    df.loc[df['inventory_ratio'] > 1.2, 'rule_price'] *= 0.93

# Round
df['rule_price'] = df['rule_price'].round(2)

# Static revenue
df['static_revenue'] = df['Price'] * df['Units Sold']

# Elasticity-based expected units (fallback -0.5)
if 'elasticity' not in df.columns:
    # Try to estimate elasticity from pct change history if possible, else fill NA
    df['elasticity'] = np.nan

df['pct_price_change'] = (df['rule_price'] - df['Price']) / (df['Price'] + 1e-6)

df['elasticity_filled'] = df['elasticity'].fillna(-0.5)
# Clip to reasonable range
df['elasticity_filled'] = df['elasticity_filled'].clip(-5, 0.5)

df['pct_units_change_est'] = df['elasticity_filled'] * df['pct_price_change']
df['pct_units_change_est'] = df['pct_units_change_est'].clip(-0.7, 2.0)

df['expected_units'] = (df['Units Sold'] * (1 + df['pct_units_change_est'])).clip(lower=0)
df['expected_units'] = df['expected_units'].round().astype(int)

# Rule revenue
df['rule_revenue'] = df['rule_price'] * df['expected_units']

# Row-level uplift
df['revenue_lift'] = df['rule_revenue'] - df['static_revenue']

# Totals
total_static_revenue = df['static_revenue'].sum()
total_rule_revenue = df['rule_revenue'].sum()
revenue_uplift = total_rule_revenue - total_static_revenue
revenue_uplift_pct = (revenue_uplift / total_static_revenue) * 100

# Comparison table
comp = pd.DataFrame({
    'Metric': ['Total Static Revenue', 'Total Rule Revenue', 'Revenue Lift', 'Revenue Lift %'],
    'Value': [total_static_revenue, total_rule_revenue, revenue_uplift, revenue_uplift_pct]
})

comp.to_csv(OUTPUT_CSV, index=False)

print('===== RESULTS =====')
print(f'Total Static Revenue     : {total_static_revenue:,.2f}')
print(f'Total Rule Revenue       : {total_rule_revenue:,.2f}')
print(f'Revenue Lift             : {revenue_uplift:,.2f}')
print(f'Revenue Lift (%)         : {revenue_uplift_pct:.2f}%')
print('\nSaved comparison to:', OUTPUT_CSV)
