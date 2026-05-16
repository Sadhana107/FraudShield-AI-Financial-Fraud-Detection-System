import numpy as np

def add_dashboard_features(df):

    # FRAUD LABEL

    df["Fraud_Label"] = df["Class"].replace({
        0: "Normal",
        1: "Fraud"
    })

    # HOUR FEATURE

    df["Hour"] = (
        df["Time"] // 3600
    ) % 24

    # RISK AMOUNT LEVEL

    def amount_risk(amount):

        if amount >= 2000:
            return "HIGH"

        elif amount >= 500:
            return "MEDIUM"

        else:
            return "LOW"

    df["Risk_Amount_Level"] = (
        df["Amount"]
        .apply(amount_risk)
    )

    # TRANSACTION TYPE

    np.random.seed(42)

    df["Transaction_Type"] = np.random.choice(
        [
            "Online",
            "POS",
            "ATM",
            "Wire",
            "Crypto"
        ],
        len(df)
    )

    # RISK ZONE

    df["Risk_Zone"] = np.random.choice(
        [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        len(df)
    )

    return df