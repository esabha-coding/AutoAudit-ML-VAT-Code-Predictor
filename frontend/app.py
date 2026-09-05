import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib

# ── Load Model ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")


@st.cache_resource
def load_model():
    tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
    le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    return tfidf, model, le


tfidf, model, le = load_model()

VAT_LABELS = {
    "standard_rate": "Standard Rate (20%)",
    "reduced_rate": "Reduced Rate (5%)",
    "zero_rated": "Zero Rated (0%)",
    "exempt": "Exempt",
    "outside_scope": "Outside Scope",
}

VAT_CODE_COLORS = {
    "standard_rate": "#2F6FED",
    "reduced_rate": "#FF8A3D",
    "zero_rated": "#24B47E",
    "exempt": "#8B9AA9",
    "outside_scope": "#1E2A38",
}

VAT_INFO = {
    "standard_rate": "Applies to most goods and services — electronics, software, restaurants, mobile contracts.",
    "reduced_rate": "Applies to domestic energy — British Gas, E.ON, Octopus Energy, water bills.",
    "zero_rated": "Applies to cold food, books, public transport — Tesco, Trainline, Amazon Kindle.",
    "exempt": "Applies to bank charges, insurance, postage — no VAT charged or reclaimable.",
    "outside_scope": "Not subject to VAT — HMRC payments, payroll, inter-account transfers."
}


# ── Helpers ─────────────────────────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_single(description):
    cleaned = clean_text(description)
    vec = tfidf.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    top_idx = int(np.argmax(probs))
    predicted_class = str(le.classes_[top_idx])
    confidence = float(probs[top_idx])
    prob_dict = {str(cls): float(p) for cls, p in zip(le.classes_, probs)}
    return predicted_class, confidence, prob_dict


def make_kpi_card(title, value, delta="", accent="#2F6FED"):
    st.markdown(
        f"""
        <div style="padding:18px 18px 16px 18px; border-radius:18px; background:linear-gradient(135deg, rgba(15,23,42,0.94), rgba(30,41,59,0.88)); border:1px solid rgba(148,163,184,0.18); box-shadow:0 18px 36px rgba(15,23,42,0.20); margin-bottom:12px; backdrop-filter: blur(10px);">
            <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.12em; font-weight:800;">{title}</div>
            <div style="font-size:30px; color:#f8fafc; font-weight:800; margin-top:10px; line-height:1.1;">{value}</div>
            <div style="font-size:12px; color:{accent}; margin-top:8px; font-weight:700; letter-spacing:0.03em;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, subtitle):
    st.markdown(
        f"""
        <div style="margin: 0.2rem 0 1rem 0; padding: 0.8rem 1rem 0.9rem 1rem; background: rgba(15,23,42,0.35); border:1px solid rgba(148,163,184,0.15); border-radius:14px;">
            <div style="font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:#a5b4cf; font-weight:800;">Section</div>
            <div style="font-size:2rem; font-weight:800; color:#f8fafc; margin-top:0.3rem;">{title}</div>
            <div style="font-size:0.95rem; color:#cbd5e1; margin-top:0.35rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoAudit-ML VAT Predictor",
    page_icon="🇬🇧",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(168,85,247,0.18), transparent 34%),
                radial-gradient(circle at top right, rgba(59,130,246,0.14), transparent 28%),
                linear-gradient(135deg, #070d17 0%, #0e1728 34%, #111827 100%);
            color: #e5edf8;
        }
        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }
        div[data-testid="stSidebar"] {
            background: rgba(15,23,42,0.8);
            border-right: 1px solid rgba(148,163,184,0.14);
            backdrop-filter: blur(16px);
            width: 320px !important;
            min-width: 320px !important;
            max-width: 320px !important;
        }
        div[data-testid="stSidebar"] > div {
            padding: 1rem 1rem 1.5rem 1rem;
        }
        .sidebar-content {
            background: transparent;
        }
        div[data-testid="stMetricContainer"] {
            background: rgba(15,23,42,0.74);
            border: 1px solid rgba(148,163,184,0.18);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 20px 40px rgba(2,6,23,0.28);
        }
        .stButton > button {
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #8b5cf6, #2f6fed);
            color: #f8fafc;
            font-weight: 800;
            letter-spacing: 0.02em;
            padding: 0.68rem 1.4rem;
            box-shadow: 0 12px 28px rgba(96, 102, 242, 0.32);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #7c4dff, #255de3);
            color: white;
        }
        .stTextInput > div > div > input,
        .stFileUploader > div {
            border-radius: 12px;
            border: 1px solid rgba(148,163,184,0.26);
            background: rgba(15,23,42,0.58);
            color: #e2e8f0;
        }
        .stTextInput input::placeholder {
            color: #94a3b8;
        }
        .stFileUploader > div {
            border: 1px dashed rgba(148,163,184,0.38);
            background: rgba(15,23,42,0.38);
        }
        h1 {
            font-size: 2.5rem !important;
            font-weight: 900 !important;
            letter-spacing: -0.05em;
            color: #f8fafc;
            margin: 0;
        }
        h2, h3, h4 {
            color: #f8fafc;
            font-weight: 700;
        }
        .section-label {
            font-size: 11px;
            color: #a5b4cf;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
        }
        .stDataFrame, .stTable {
            background: rgba(15,23,42,0.55);
            border-radius: 14px;
        }
        .stAlert {
            background: rgba(15,23,42,0.35);
            border: 1px solid rgba(148,163,184,0.15);
            color: #dfe7f5;
        }
        .stCaption {
            color: #cbd5e1 !important;
        }
        p, li, div, span, label {
            color: #e2e8f0;
        }
        th, td {
            color: #edf2ff;
        }
        .block-container .stMarkdown {
            color: #edf2ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 0.5rem 0 1.15rem 0; border-bottom: 1px solid rgba(148,163,184,0.15); margin-bottom: 1.1rem;">
        <div class="section-label">MTD COMPLIANCE PLATFORM</div>
        <h1 style="margin-top:0.42rem;">AutoAudit-ML VAT Intelligence</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("AI-powered HMRC VAT classification for UK bank transactions • Executive-grade audit decision support")

page = st.sidebar.radio("Navigation", ["Single Transaction", "Batch Upload", "VAT Reference"])

with st.sidebar:
    st.markdown("---")
    st.markdown("### Executive Summary")
    st.markdown("- Model: XGBoost + TF-IDF")
    st.markdown("- Training samples: 5,187")
    st.markdown("- Macro F1: 0.9994")
    st.markdown("- Classes: 5 VAT groups")
    st.markdown("---")
    st.markdown("### Value Delivered")
    st.markdown("- Faster audit review")
    st.markdown("- Cleaner tax coding")
    st.markdown("- Decision-ready explanations")


if page == "Single Transaction":
    render_section_title(
        "Single Transaction Classification",
        "Enter a transaction description and review the VAT prediction, confidence, and probability breakdown."
    )

    transaction_desc = st.text_input(
        "Enter a transaction description",
        placeholder="e.g. TESCO STORES 3421 LONDON",
    )

    st.caption("Examples: TESCO STORES • AMAZON PRIME • HMRC SHIPLEY • BRITISH GAS • TRAINLINE • BARCLAYS BANK CHARGE")

    if st.button("Predict VAT Category", type="primary"):
        if not transaction_desc.strip():
            st.warning("Please enter a transaction description.")
        else:
            with st.spinner("Running VAT classification..."):
                predicted_class, confidence, prob_dict = predict_single(transaction_desc)

            label = VAT_LABELS.get(predicted_class, predicted_class)
            info = VAT_INFO.get(predicted_class, "")
            color = VAT_CODE_COLORS.get(predicted_class, "#2F6FED")

            col1, col2, col3 = st.columns(3)
            with col1:
                make_kpi_card("Predicted VAT Code", label, "Classification result", color)
            with col2:
                make_kpi_card("Confidence", f"{confidence:.1%}", "Model certainty", color)
            with col3:
                make_kpi_card("HMRC Group", predicted_class.replace("_", " ").title(), "Tax classification", color)

            prob_df = pd.DataFrame({
                "VAT Code": [VAT_LABELS.get(k, k) for k in prob_dict.keys()],
                "Probability": list(prob_dict.values()),
                "Key": list(prob_dict.keys()),
            }).sort_values("Probability", ascending=False)

            pie_fig = px.pie(
                prob_df,
                names="VAT Code",
                values="Probability",
                color="Key",
                color_discrete_map={key: VAT_CODE_COLORS.get(key, "#2F6FED") for key in VAT_CODE_COLORS},
                hole=0.45,
            )
            pie_fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="<b>%{label}</b><br>Probability: %{value:.2%}<extra></extra>")
            pie_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="v", yanchor="middle", xanchor="left", x=1.02),
                height=360,
            )

            bar_fig = px.bar(
                prob_df,
                x="Probability",
                y="VAT Code",
                orientation="h",
                color="Probability",
                color_continuous_scale="Blues",
                range_color=(0, 1),
                text="Probability",
            )
            bar_fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            bar_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                xaxis=dict(range=[0, 1.0]),
                showlegend=False,
            )

            left_col, right_col = st.columns([1.1, 1.2])
            with left_col:
                st.markdown("### Probability Distribution")
                st.plotly_chart(pie_fig, use_container_width=True)
            with right_col:
                st.markdown("### Confidence Ranking")
                st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("### Business Interpretation")
            st.info(f"{label} is the most likely VAT category for this transaction. Suggested explanation: {info}")

elif page == "Batch Upload":
    render_section_title(
        "Batch VAT Audit",
        "Upload a CSV file to classify multiple transactions and review trends, totals, and confidence scores."
    )
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### File Preview")
        st.dataframe(df.head(10), use_container_width=True)

        desc_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["desc", "transaction", "detail", "narration", "merchant"]):
                desc_col = col
                break
        if desc_col is None:
            desc_col = df.columns[0]

        st.info(f"Using the **{desc_col}** column for prediction.")

        if st.button("Run Batch Prediction", type="primary"):
            with st.spinner("Processing all transactions..."):
                descriptions = df[desc_col].fillna("").tolist()
                cleaned = [clean_text(d) for d in descriptions]
                vectorized = tfidf.transform(cleaned)
                probs_matrix = model.predict_proba(vectorized)
                top_indices = np.argmax(probs_matrix, axis=1)

                df["predicted_vat_code"] = [str(le.classes_[i]) for i in top_indices]
                df["vat_label"] = df["predicted_vat_code"].map(VAT_LABELS)
                df["confidence"] = [round(float(probs_matrix[i, top_indices[i]]), 4) for i in range(len(descriptions))]

            total = len(df)
            avg_conf = df["confidence"].mean()
            top_code = df["predicted_vat_code"].mode().iloc[0]

            c1, c2, c3 = st.columns(3)
            with c1:
                make_kpi_card("Transactions", f"{total}", "Rows processed", "#2F6FED")
            with c2:
                make_kpi_card("Top VAT Code", VAT_LABELS.get(top_code, top_code), "Most frequent class", "#24B47E")
            with c3:
                make_kpi_card("Average Confidence", f"{avg_conf:.1%}", "Model certainty", "#FF8A3D")

            distribution = df["predicted_vat_code"].value_counts().reset_index()
            distribution.columns = ["VAT Code", "Count"]
            distribution["VAT Label"] = distribution["VAT Code"].map(VAT_LABELS)

            dist_fig = px.bar(
                distribution,
                x="VAT Label",
                y="Count",
                color="Count",
                color_continuous_scale="Viridis",
                text="Count",
            )
            dist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=340,
            )
            dist_fig.update_traces(textposition="outside")

            conf_fig = px.histogram(
                df,
                x="confidence",
                nbins=20,
                color_discrete_sequence=["#2F6FED"],
            )
            conf_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
            )

            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("### VAT Distribution")
                st.plotly_chart(dist_fig, use_container_width=True)
            with right_col:
                st.markdown("### Confidence Distribution")
                st.plotly_chart(conf_fig, use_container_width=True)

            display_columns = [desc_col, "predicted_vat_code", "vat_label", "confidence"]
            st.markdown("### Prediction Results")
            st.dataframe(df[display_columns], use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV Results", csv, file_name="vat_predictions.csv", mime="text/csv")

else:
    render_section_title(
        "HMRC VAT Code Reference",
        "Use this reference table to understand the tax treatment for each VAT category and common examples."
    )
    st.markdown(
        """
        <div style="background: rgba(248,250,252,0.96); border:1px solid rgba(71,85,105,0.24); border-radius:18px; padding:18px 20px; box-shadow: 0 12px 26px rgba(15,23,42,0.12);">
            <table style="width:100%; border-collapse:collapse; font-size:16px; line-height:1.5; color:#0f172a;">
                <thead>
                    <tr>
                        <th style="text-align:left; padding:14px 12px; border-bottom:2px solid #cbd5e1; color:#0f172a; font-size:17px; font-weight:800;">VAT Code</th>
                        <th style="text-align:left; padding:14px 12px; border-bottom:2px solid #cbd5e1; color:#0f172a; font-size:17px; font-weight:800;">Rate</th>
                        <th style="text-align:left; padding:14px 12px; border-bottom:2px solid #cbd5e1; color:#0f172a; font-size:17px; font-weight:800;">Examples</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="background: rgba(255,255,255,0.25);">
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:700;">🔵 Standard Rate</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:600;">20%</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#334155;">Electronics, SaaS, restaurants, mobile contracts</td>
                    </tr>
                    <tr>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:700;">🟠 Reduced Rate</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:600;">5%</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#334155;">British Gas, E.ON, Octopus Energy, water bills</td>
                    </tr>
                    <tr style="background: rgba(255,255,255,0.25);">
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:700;">🟢 Zero Rated</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:600;">0%</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#334155;">Tesco, Sainsbury’s, Trainline, Amazon Kindle</td>
                    </tr>
                    <tr>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:700;">⚪ Exempt</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#0f172a; font-weight:600;">N/A</td>
                        <td style="padding:14px 12px; border-bottom:1px solid #e2e8f0; color:#334155;">Bank charges, insurance, Royal Mail, postage</td>
                    </tr>
                    <tr style="background: rgba(255,255,255,0.25);">
                        <td style="padding:14px 12px; color:#0f172a; font-weight:700;">⬛ Outside Scope</td>
                        <td style="padding:14px 12px; color:#0f172a; font-weight:600;">N/A</td>
                        <td style="padding:14px 12px; color:#334155;">HMRC payments, payroll, internal transfers</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Built by Saba Ijaz** — AI Automation Builder & Executive Accountant | MTD Systems & Tax Intelligence")