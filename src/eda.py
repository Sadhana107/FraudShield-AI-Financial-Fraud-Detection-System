import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------
# LOAD DATA
# --------------------------------------

df = pd.read_csv("data/creditcard.csv")

# --------------------------------------
# FRAUD DISTRIBUTION
# --------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x=df["Class"]
)

plt.title("Fraud vs Non-Fraud Transactions")

plt.xticks(
    [0,1],
    ["Non-Fraud", "Fraud"]
)

plt.savefig(
    "outputs/fraud_distribution.png"
)

plt.show()

# --------------------------------------
# TRANSACTION AMOUNT DISTRIBUTION
# --------------------------------------

plt.figure(figsize=(10,5))

sns.histplot(
    df["Amount"],
    bins=50
)

plt.title("Transaction Amount Distribution")

plt.savefig(
    "outputs/amount_distribution.png"
)

plt.show()