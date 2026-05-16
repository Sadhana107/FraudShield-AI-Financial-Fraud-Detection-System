import pandas as pd
import joblib

# LOAD MODEL

model = joblib.load(
    "models/fraud_model.pkl"
)

def predict_transaction(input_data):

    input_df = pd.DataFrame([input_data])

    probability = (
        model.predict_proba(input_df)[0][1]
    )

    prediction = (
        model.predict(input_df)[0]
    )

    # RISK LOGIC

    if probability >= 0.80:

        risk = "HIGH RISK"

        decision = "AUTO BLOCK"

    elif probability >= 0.40:

        risk = "MEDIUM RISK"

        decision = "MANUAL REVIEW"

    else:

        risk = "LOW RISK"

        decision = "SAFE APPROVAL"

    return {

        "prediction":
        int(prediction),

        "fraud_probability":
        round(probability * 100, 2),

        "risk_level":
        risk,

        "decision":
        decision
    }