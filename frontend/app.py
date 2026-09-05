import os
import sys
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

# ── Load Model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")

@st.cache_resource
def load_model():
    tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
    le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    return tfidf, model, le

tfidf, model, le = load_model()

VAT_LABELS = {
    "standard_rate": "🔵 Standard Rate (20%)",
    "reduced_rate": "🟠 Reduced Rate (5%)",
    "zero_rated": "🟢 Zero Rated (0%)",
    "exempt": "⚪ Exempt",
    "outside_scope": "⬛ Outside Scope"
}

VAT_INFO = {
    "standard_rate": "Applies to most goods and services — electronics, software, restaurants, mobile contracts.",
    "reduced_rate": "Applies to domestic energy — British Gas, E.ON, Octopus Energy, water bills.",
    "zero_rated": "Applies to cold food, books, public transport — Tesco, Trainline, Amazon Kindle.",
    "exempt": "Applies to bank charges, insurance, postage — no VAT charged or reclaimable.",
    "outside_scope": "Not subject to VAT — HMRC payments, payroll, inter-account transfers."
}

# ── Helper ─────────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_single(description):
    cleaned = clean_text(description)
    vec = tfidf.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    top_idx = int(np.argmax(probs))
    predicted_class = str(le.classes_[top_idx])
    confidence = float(probs[top_idx])
    prob_dict = {str(cls): float(p) for cls, p in zip(le.classes_, probs)}
    return predicted_class, confidence, prob_dict

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoAudit-ML VAT Predictor",
    page_icon="🇬🇧",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
div[data-testid="stMetricContainer"] { background: #ffffff; border: 1px solid #e8e8e8; padding: 0.9rem; border-radius: 0.85rem; }
h1 { font-size: 2.1rem !important; font-weight: 700 !important; letter-spacing: -0.04em; }
.stButton > button { border: 1px solid #d9d9d9; border-radius: 0.6rem; color: #1f1f1f; background: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("🇬🇧 AutoAudit-ML: MTD VAT Code Predictor")
st.caption("Automated HMRC VAT classification for UK bank transactions | MTD ITSA April 2026")

# ── Sidebar ────────────────────────────────────────────────────────────────────
page = st.sidebar.radio("Navigation", ["Single Transaction", "Batch Upload", "VAT Reference"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Algorithm: XGBoost + TF-IDF")
st.sidebar.markdown("- Training samples: 5,187")
st.sidebar.markdown("- Macro F1: 0.9994")
st.sidebar.markdown("- Classes: 5 VAT codes")

# ── Single Transaction ─────────────────────────────────────────────────────────
if page == "Single Transaction":
    st.header("Single Transaction Classifier")

    transaction_desc = st.text_input(
        "Enter Bank Transaction Description:",
        placeholder="e.g., TESCO STORES 3421 LONDON",
    )

    st.caption("Try: TESCO STORES | AMAZON PRIME | HMRC SHIPLEY | BRITISH GAS | TRAINLINE | BARCLAYS BANK CHARGE")

    if st.button("🔍 Predict VAT Category", type="primary"):
        if transaction_desc.strip():
            with st.spinner("Analyzing transaction..."):
                predicted_class, confidence, prob_dict = predict_single(transaction_desc)

            label = VAT_LABELS.get(predicted_class, predicted_class)
            info = VAT_INFO.get(predicted_class, "")

            st.success("Prediction complete")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VAT Code", label)
            with col2:
                st.metric("Confidence", f"{confidence:.1%}")
            with col3:
                st.metric("HMRC Class", predicted_class.replace("_", " ").title())

            prob_df = pd.DataFrame({
                "VAT Code": [VAT_LABELS.get(k, k) for k in prob_dict.keys()],
                "Probability": list(prob_dict.values())
            }).sort_values("Probability", ascending=True)

            fig = px.bar(
                prob_df, x="Probability", y="VAT Code",
                orientation="h", color="Probability",
                color_continuous_scale="Blues", range_color=(0, 1),
            )
            fig.update_layout(
                paper_bgcolor="white", plot_bgcolor="white",
                height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, xaxis=dict(range=[0, 1.05]),
            )
            st.subheader("Probability Breakdown")
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 HMRC Explanation"):
                st.write(info)
        else:
            st.warning("Please enter a transaction description.")

# ── Batch Upload ───────────────────────────────────────────────────────────────
elif page == "Batch Upload":
    st.header("Batch VAT Audit & Analytics")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        desc_col = None
        for col in df.columns:
            if any(x in col.lower() for x in ["desc", "transaction", "detail", "narration", "merchant"]):
                desc_col = col
                break
        if desc_col is None:
            desc_col = df.columns[0]

        st.info(f"Using column: **{desc_col}** for predictions")

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing all transactions..."):
                descriptions = df[desc_col].fillna("").tolist()
                cleaned = [clean_text(d) for d in descriptions]
                vectorized = tfidf.transform(cleaned)
                probs_matrix = model.predict_proba(vectorized)
                top_indices = np.argmax(probs_matrix, axis=1)

                df["predicted_vat_code"] = [str(le.classes_[i]) for i in top_indices]
                df["vat_label"] = df["predicted_vat_code"].map(VAT_LABELS)
                df["confidence"] = [round(float(probs_matrix[i, top_indices[i]]), 4) for i in range(len(descriptions))]

            st.success("Batch processing complete")

            total = len(df)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transactions", total)
            with col2:
                st.metric("Top VAT Code", df["predicted_vat_code"].mode().iloc[0])
            with col3:
                st.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

            chart_df = df["predicted_vat_code"].value_counts().reset_index()
            chart_df.columns = ["VAT Code", "Count"]
            fig = px.bar(chart_df, x="VAT Code", y="Count", color="Count", color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.subheader("VAT Distribution")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Results")
            st.dataframe(df[["description" if "description" in df.columns else desc_col, "predicted_vat_code", "vat_label", "confidence"]], use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv, "vat_predictions.csv", "text/csv")

# ── VAT Reference ──────────────────────────────────────────────────────────────
elif page == "VAT Reference":
    st.header("📋 HMRC VAT Code Reference")

    st.markdown("""
    | VAT Code | Rate | Common Examples |
    |---|---|---|
    | 🔵 Standard Rate | 20% | Electronics, SaaS, Restaurants, Mobile, Hotels |
    | 🟠 Reduced Rate | 5% | British Gas, E.ON, Octopus, Water utilities |
    | 🟢 Zero Rated | 0% | Tesco, Sainsburys, Trainline, Amazon Kindle, Waterstones |
    | ⚪ Exempt | N/A | Bank charges, Insurance, Royal Mail, NHS |
    | ⬛ Outside Scope | N/A | HMRC payments, Payroll, Inter-account transfers |
    """)

    st.markdown("---")
    st.markdown("**Built by Saba Ijaz** — AI Automation Builder & Executive Accountant | Xero Advisor Certified | Making Tax Digital")