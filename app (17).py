
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "fraud_attention_model.keras"
    )

@st.cache_resource
def load_scaler():
    with open(
        "scaler.pkl",
        "rb"
    ) as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

st.title("💳 Fraud Detection Dashboard")

uploaded = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded:

    df = pd.read_csv(uploaded)

    st.subheader("Uploaded Transactions")
    st.dataframe(df.head())

    try:

        data = scaler.transform(df)

        X = np.expand_dims(
            data,
            axis=0
        )

        prediction = model.predict(X)

        fraud_probability = float(
            prediction[0][0]
        )

        st.metric(
            "Fraud Probability",
            round(
                fraud_probability,
                4
            )
        )

        if fraud_probability > 0.8:
            st.error(
                "High Risk Transaction"
            )

        elif fraud_probability > 0.5:
            st.warning(
                "Medium Risk Transaction"
            )

        else:
            st.success(
                "Low Risk Transaction"
            )

    except Exception as e:
        st.error(str(e))
