import streamlit as st
import pandas as pd
import joblib
@st.cache_resource
def load_model():
    return joblib.load("xgboost_pcn1_model.pkl")

model = load_model()

st.set_page_config(page_title="Dự đoán Sạch Sỏi", layout="centered")

st.title("🔮 Dự Đoán Kết Quả PCNL")
st.markdown("**Nhập thông tin bệnh nhân để dự đoán xác suất sạch sỏi**")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
    hu = st.number_input("HU (Hounsfield)", min_value=100, max_value=2000, value=800)
    stone_size = st.number_input("Kích thước sỏi (mm)", min_value=5.0, max_value=50.0, value=15.0)
    hydro = st.selectbox("Mức độ giãn đài bể thận", [0, 1, 2, 3, 4])

with col2:
    stone_count_str = st.selectbox("Số lượng sỏi", ["1", "2", "3", "4", ">3"])
    stone_count = 4 if stone_count_str == ">3" else int(stone_count_str)
    
    urine_ph = st.number_input("Urine pH", min_value=4.5, max_value=8.5, value=6.0, step=0.1)
    distance = st.number_input("Distance to UPJ (mm)", min_value=0.0, max_value=50.0, value=5.0)
    location = st.selectbox("Vị trí sỏi", ["Lower", "Middle", "Upper"])

if st.button("🚀 Dự Đoán", type="primary", use_container_width=True):
    
    input_df = pd.DataFrame({
        'BMI': [bmi],
        'HU': [hu],
        'Stone_Size_mm': [stone_size],
        'Hydronephrosis_Grade': [hydro],
        'Stone_Count': [stone_count],
        'Urine_pH': [urine_ph],
        'Distance_to_UPJ_mm': [distance],
        'Stone_Location_Lower': [1 if location == "Lower" else 0],
        'Stone_Location_Middle': [1 if location == "Middle" else 0],
    })

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success("✅ **Dự đoán: SẠCH SỎI**")
    else:
        st.error("❌ **Dự đoán: KHÔNG SẠCH SỎI**")
    
    st.metric("Xác suất sạch sỏi", f"{proba*100:.1f}%")

st.caption("Ứng dụng hỗ trợ tham khảo - Không thay thế chẩn đoán bác sĩ")
