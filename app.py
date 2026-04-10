import streamlit as st
import pickle
import numpy as np

st.title("Dự đoán sỏi thận PCNL")

# Load model
model = pickle.load(open("xgboost_pcnl_model.pkl", "rb"))

# Input demo (bạn sửa theo feature thật sau)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")

if st.button("Dự đoán"):
    input_data = np.array([[feature1, feature2, feature3]])
    result = model.predict(input_data)
    st.success(f"Kết quả: {result}")
