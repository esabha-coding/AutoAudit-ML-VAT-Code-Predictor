import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import get_single_prediction, get_batch_prediction

st.set_page_config(page_title="AutoAudit-ML | MTD VAT Engine", layout="wide")

st.title("🇬🇧 AutoAudit-ML: MTD VAT Code Predictor")
st.markdown("Automated HMRC VAT classification & Compliance Analytics for UK bank statements.")

tab1, tab2 = st.tabs(["Single Transaction Audit", "Batch Multi-CSV Audit & Analytics"])

# --- TAB 1: SINGLE TRANSACTION ---
with tab1:
    st.subheader("Single Transaction Classifier")
    desc_input = st.text_input("Enter Bank Transaction Description:", "TESCO STORES 3421 LONDON")
    
    if st.button("Predict VAT Category"):
        res = get_single_prediction(desc_input)
        if "error" in res:
            st.error(f"Cannot reach FastAPI server. Ensure backend is running on port 8000! ({res['error']})")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted VAT Code", res["predicted_vat_code"].upper())
                st.metric("Confidence Level", f"{res['confidence'] * 100:.1f}%")
                st.info(res["explanation"])
            
            with col2:
                probs_df = pd.DataFrame(list(res["probabilities"].items()), columns=["VAT Code", "Probability"])
                fig = px.bar(probs_df, x="Probability", y="VAT Code", orientation="h", title="Class Probabilities", color="Probability", color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: MULTI-FILE & DATE-WISE ANALYTICS ---
with tab2:
    st.subheader("Upload Bank Statement CSVs")
    uploaded_files = st.file_uploader("Upload one or multiple bank statement CSVs", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        df_list = []
        for file in uploaded_files:
            temp_df = pd.read_csv(file)
            temp_df["source_file"] = file.name
            df_list.append(temp_df)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        st.write(f"Combined Preview ({len(combined_df)} total records from {len(uploaded_files)} file(s)):", combined_df.head(5))
        
        # Smart column auto-detection prioritizing raw_description
        cols = list(combined_df.columns)
        default_desc_idx = cols.index("raw_description") if "raw_description" in cols else (cols.index("description") if "description" in cols else 0)
        default_date_idx = (cols.index("date") + 1) if "date" in cols else ((cols.index("transaction_date") + 1) if "transaction_date" in cols else 0)

        col_select1, col_select2 = st.columns(2)
        with col_select1:
            desc_col = st.selectbox("Select description column:", cols, index=default_desc_idx)
        with col_select2:
            date_col = st.selectbox("Select date column (optional):", [None] + cols, index=default_date_idx)

        if st.button("Process Batch Categorization & Generate Analytics"):
            descriptions = combined_df[desc_col].astype(str).tolist()
            res = get_batch_prediction(descriptions)
            
            if "error" in res:
                st.error("Backend error during batch categorization.")
            else:
                combined_df["Predicted_VAT_Code"] = [r["predicted_vat_code"] for r in res["results"]]
                combined_df["Confidence"] = [r["confidence"] for r in res["results"]]
                
                st.success("Batch categorization and compliance analysis complete!")
                
                # --- METRICS ROW ---
                st.divider()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Transactions", f"{len(combined_df):,}")
                m2.metric("Avg Confidence", f"{combined_df['Confidence'].mean() * 100:.1f}%")
                m3.metric("Standard Rate Count", f"{(combined_df['Predicted_VAT_Code'] == 'standard_rate').sum():,}")
                m4.metric("Zero/Exempt Count", f"{(combined_df['Predicted_VAT_Code'].isin(['zero_rated', 'exempt'])).sum():,}")
                
                st.divider()
                
                # --- VISUALIZATIONS ---
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    vat_counts = combined_df["Predicted_VAT_Code"].value_counts().reset_index()
                    vat_counts.columns = ["VAT Code", "Count"]
                    fig_pie = px.pie(vat_counts, names="VAT Code", values="Count", title="VAT Code Distribution Breakdown", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with chart_col2:
                    if date_col and date_col in combined_df.columns:
                        combined_df["parsed_date"] = pd.to_datetime(combined_df[date_col], errors="coerce")
                        date_df = combined_df.dropna(subset=["parsed_date"]).sort_values("parsed_date")
                        
                        if not date_df.empty:
                            trend_df = date_df.groupby([date_df["parsed_date"].dt.date, "Predicted_VAT_Code"]).size().reset_index(name="Transaction_Count")
                            fig_trend = px.bar(trend_df, x="parsed_date", y="Transaction_Count", color="Predicted_VAT_Code", title="Date-Wise Transaction Volume by VAT Category", barmode="stack")
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.info("Could not parse valid dates for trend plotting.")
                    else:
                        st.info("Select a date column above to visualize date-wise trends.")

                # --- DATA TABLE & DOWNLOAD ---
                st.subheader("MTD Audit-Ready Transaction Dataset")
                st.dataframe(combined_df, use_container_width=True)
                
                csv_data = combined_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Consolidated MTD Categorized CSV", csv_data, "mtd_vat_consolidated.csv", "text/csv")