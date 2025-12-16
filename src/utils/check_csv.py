import pandas as pd

# Quick check of CSV structure
csv_path = 'data/images/nih_labels_2017.csv'
df = pd.read_csv(csv_path, nrows=5)

print("CSV Columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
