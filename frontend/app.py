import streamlit as st
from utils.api_client import predict_transaction, predict_batch

st.set_page_config(
    page_title="AutoAudit-ML VAT Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("🏦 AutoAudit-ML: MTD VAT Code Predictor")
st.markdown("Automated HMRC VAT classification & Compliance Analytics for UK bank statements.")

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Single Transaction", "Batch Upload"])

if page == "Single Transaction":
    st.header("Single Transaction Classifier")
    
    transaction_desc = st.text_input(
        "Enter Bank Transaction Description:",
        placeholder="e.g., TESCO STORES 3421 LONDON"
    )
    
    amount = st.number_input("Amount (optional):", value=None, step=0.01)
    
    if st.button("Predict VAT Category"):
        if transaction_desc:
            with st.spinner("Analyzing transaction..."):
                result = predict_transaction(transaction_desc, amount)
                if result:
                    st.success("✅ Prediction Complete")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VAT Code", result.get("vat_code", "N/A"))
                    with col2:
                        st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                    with col3:
                        st.metric("Category", result.get("category", "N/A"))
        else:
            st.warning("⚠️ Please enter a transaction description")

elif page == "Batch Upload":
    st.header("Batch Multi-CSV Audit & Analytics")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file:
        if st.button("Process Batch"):
            with st.spinner("Processing transactions..."):
                result = predict_batch(uploaded_file)
                if result:
                    st.success("✅ Batch Processing Complete")
                    st.json(result)