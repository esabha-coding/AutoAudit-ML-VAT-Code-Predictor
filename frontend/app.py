import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.api_client import predict_transaction, predict_batch

st.set_page_config(
    page_title="AutoAudit-ML VAT Predictor",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #ffffff; }
    .block-container { padding-top: 0.8rem; padding-bottom: 1rem; }
    div[data-testid="stSidebar"] { background: #f7f7f7; border-right: 0px solid #ececec; }
    div[data-testid="stMetricContainer"] { background: #ffffff; border: 1px solid #e8e8e8; padding: 0.9rem; border-radius: 0.85rem; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
    h1 { font-size: 2.1rem !important; font-weight: 700 !important; letter-spacing: -0.04em; margin-bottom: 0.2rem !important; }
    h2, h3 { letter-spacing: -0.02em; }
    .stButton > button { border: 1px solid #d9d9d9; border-radius: 0.6rem; color: #1f1f1f; background: #ffffff; }
    .stTextInput > div > div > input, .stNumberInput > div > div > input { border-radius: 0.6rem; border: 1px solid #d9d9d9; }
    .top-banner { background: #fbfbfb; border: 1px solid #ededed; border-radius: 0.8rem; padding: 0.65rem 0.9rem; margin-bottom: 0.8rem; }
    .brand-text { font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #666; }
    .sidebar-title { font-size: 0.74rem; letter-spacing: 0.1em; text-transform: uppercase; color: #666; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-banner">
        <div class="brand-text">VAT compliance analytics</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("AutoAudit-ML: MTD VAT Code Predictor")
st.caption("Automated HMRC VAT classification and compliance analytics for UK bank transactions.")

st.sidebar.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Single Transaction", "Batch Upload"], label_visibility="collapsed")

if page == "Single Transaction":
    st.header("Single Transaction Classifier")

    transaction_desc = st.text_input(
        "Enter Bank Transaction Description:",
        placeholder="e.g., TESCO STORES 3421 LONDON",
    )

    amount = st.number_input("Amount (optional):", min_value=0.0, value=0.0, step=0.01)
    amount_value = None if amount == 0.0 else amount

    if st.button("Predict VAT Category"):
        if transaction_desc:
            with st.spinner("Analyzing transaction..."):
                result = predict_transaction(transaction_desc, amount_value)
                if result:
                    st.success("Prediction complete")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VAT Code", result.get("vat_code", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col3:
                        st.metric("Category", result.get("category", "N/A"))

                    probabilities = result.get("probabilities", {})
                    if probabilities:
                        prob_df = pd.DataFrame(
                            {
                                "VAT Code": list(probabilities.keys()),
                                "Probability": [float(v) for v in probabilities.values()],
                            }
                        ).sort_values("Probability", ascending=False)

                        fig = px.bar(
                            prob_df,
                            x="Probability",
                            y="VAT Code",
                            orientation="h",
                            color="Probability",
                            color_continuous_scale="Greys",
                            range_color=(0, 1),
                        )
                        fig.update_layout(
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            height=360,
                            margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=False,
                            xaxis=dict(range=[0, 1.05]),
                        )
                        st.subheader("Probability Breakdown")
                        st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Explanation"):
                        st.write(result.get("explanation", "No explanation available."))
        else:
            st.warning("Please enter a transaction description.")

elif page == "Batch Upload":
    st.header("Batch VAT Audit & Analytics")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:
        if st.button("Process Batch"):
            with st.spinner("Processing transactions..."):
                result = predict_batch(uploaded_file)
                if result and "results" in result:
                    st.success("Batch processing complete")

                    records = result["results"]
                    if records:
                        summary_df = pd.DataFrame(records)
                        summary_df["vat_code"] = summary_df["vat_code"].fillna("Unknown")

                        total = len(summary_df)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Transactions", total)
                        with col2:
                            st.metric("Top VAT Code", summary_df["vat_code"].mode().iloc[0] if not summary_df.empty else "N/A")
                        with col3:
                            st.metric("Avg Confidence", f"{summary_df['confidence'].mean():.1%}" if not summary_df.empty else "0%")

                        chart_df = summary_df["vat_code"].value_counts().reset_index()
                        chart_df.columns = ["VAT Code", "Count"]
                        fig = px.bar(
                            chart_df,
                            x="VAT Code",
                            y="Count",
                            color="Count",
                            color_continuous_scale="Greys",
                        )
                        fig.update_layout(
                            paper_bgcolor="white",
                            plot_bgcolor="white",
                            height=360,
                            margin=dict(l=10, r=10, t=10, b=10),
                            showlegend=False,
                        )
                        st.subheader("Batch Distribution")
                        st.plotly_chart(fig, use_container_width=True)

                        st.subheader("Processed Results")
                        st.dataframe(summary_df[["description", "vat_code", "confidence"]], use_container_width=True)

                    st.json({"results": records})
                else:
                    st.warning("No batch results returned.")