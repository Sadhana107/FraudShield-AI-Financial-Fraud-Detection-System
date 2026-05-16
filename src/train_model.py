import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from imblearn.over_sampling import SMOTE

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

X_train = pd.read_csv("data/X_train.csv")

X_test = pd.read_csv("data/X_test.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()

y_test = pd.read_csv("data/y_test.csv").values.ravel()

# ----------------------------------------
# HANDLE IMBALANCE USING SMOTE
# ----------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

# ----------------------------------------
# TRAIN MODEL
# ----------------------------------------

model = RandomForestClassifier(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_smote,
    y_train_smote
)

# ----------------------------------------
# PREDICTIONS
# ----------------------------------------

y_pred = model.predict(X_test)

# ----------------------------------------
# EVALUATION
# ----------------------------------------

print("\nAccuracy:")
print(
    accuracy_score(y_test, y_pred)
)

print("\nClassification Report:")
print(
    classification_report(y_test, y_pred)
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(y_test, y_pred)
)

# ----------------------------------------
# SAVE MODEL
# ----------------------------------------

joblib.dump(
    model,
    "models/fraud_model.pkl"
)

print("\nModel Saved Successfully")