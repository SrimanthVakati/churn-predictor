import pandas as pd

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nHow many churned?")
print(df['Churn'].value_counts())