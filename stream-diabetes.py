import pickle
import streamlit as st
import numpy as np

# Load model
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))

# Load scaler jika ada
try:
    scaler = pickle.load(open('scaler.sav', 'rb'))
    use_scaler = True
except FileNotFoundError:
    scaler = None
    use_scaler = False

# Judul aplikasi
st.title('🩺 Prediksi Diabetes - Data Mining')

# Input
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Jumlah Kehamilan (Pregnancies)', min_value=0, max_value=17, value=1)

with col2:
    Glucose = st.number_input('Kadar Glukosa (Glucose)', min_value=0, max_value=199, value=89)

with col1:
    BloodPressure = st.number_input('Tekanan Darah (BloodPressure)', min_value=0, max_value=122, value=66)

with col2:
    SkinThickness = st.number_input('Ketebalan Kulit (SkinThickness)', min_value=0, max_value=99, value=23)

with col1:
    Insulin = st.number_input('Kadar Insulin (Insulin)', min_value=0, max_value=846, value=94)

with col2:
    BMI = st.number_input('BMI', min_value=0.0, max_value=67.1, value=28.1)

with col1:
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.078, max_value=2.42, value=0.167)

with col2:
    Age = st.number_input('Usia (Age)', min_value=21, max_value=81, value=21)

# Tombol Prediksi
if st.button('🔍 Tes Prediksi Diabetes'):
    input_data = np.array([[
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    ]])

    # Normalisasi
    if use_scaler:
        input_data = scaler.transform(input_data)

    # Prediksi
    prediction = diabetes_model.predict(input_data)

    # Tampilkan hasil
    st.write("📊 Hasil Prediksi Model (0: Tidak Diabetes, 1: Diabetes):", int(prediction[0]))

    if int(prediction[0]) == 1:
        st.success('🚨 Pasien Terkena Diabetes')
    else:
        st.info('✅ Pasien Tidak Terkena Diabetes')
