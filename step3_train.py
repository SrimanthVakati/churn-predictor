import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

df = pd.read_csv('data/clean.csv')

X = df.drop('Churn', axis=1)  # inputs (everything except Churn)
y = df['Churn']               # what we want to predict

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Train the model
model = XGBClassifier(scale_pos_weight=3, random_state=42,
                      eval_metric='logloss')
model.fit(X_train, y_train)

# Test how good it is
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nAUC Score:", round(roc_auc_score(y_test, probabilities), 3))
print("\nDetailed Report:")
print(classification_report(y_test, predictions))

# Save the model
joblib.dump(model, 'model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print("Model saved!")