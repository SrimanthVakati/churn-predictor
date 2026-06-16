import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Fix TotalCharges column (has some blank spaces instead of numbers)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)

# Remove customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Turn Churn Yes/No into 1/0
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

# Turn all text columns into numbers
df = pd.get_dummies(df, drop_first=True)

print("Clean data shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# Save it
df.to_csv('data/clean.csv', index=False)
print("\nSaved to data/clean.csv")