import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# LOAD DATA
# ----------------------------------------

df = pd.read_csv("data/creditcard.csv")

# ----------------------------------------
# FEATURES & TARGET
# ----------------------------------------

X = df.drop("Class", axis=1)

y = df["Class"]

# ----------------------------------------
# TRAIN TEST SPLIT
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# SCALE AMOUNT FEATURE
# ----------------------------------------

scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount"] = scaler.transform(
    X_test[["Amount"]]
)

# ----------------------------------------
# SAVE FILES
# ----------------------------------------

X_train.to_csv(
    "data/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/X_test.csv",
    index=False
)

y_train.to_csv(
    "data/y_train.csv",
    index=False
)

y_test.to_csv(
    "data/y_test.csv",
    index=False
)

print("Preprocessing Completed Successfully")