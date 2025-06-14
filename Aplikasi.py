import pickle
import streamlit as st
import numpy as np
import pandas as pd
import easyocr
from PIL import Image
import tempfile
import base64

# ============== BACKGROUND NAVY + TEXT KUNING ============== #
def set_background_style():
    st.markdown("""
        <style>
        .stApp {
            background-color: #0a1f44;
            color: #fdd835;
        }
        .stButton>button {
            background-color: #fdd835;
            color: #0a1f44;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stSelectbox, .stNumberInput {
            background-color: white;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)

set_background_style()

# ============== LOAD MODEL ============== #
model = pickle.load(open('prediksi_hargamobil.sav', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# ============== HEADER ============== #
st.title('🚗 Prediksi Harga Mobil Toyota')
st.image('mobil.png', use_column_width=True)

# ============== OCR: AMBIL GAMBAR PLAT ============== #
st.markdown("### 📸 Ambil Gambar Plat Nomor")
plate_image = st.camera_input("Ambil Foto Plat")

ocr_year = None  # Inisialisasi

if plate_image is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(plate_image.getbuffer())
        image_path = tmp_file.name

    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    extracted_text = " ".join(result)
    
    st.markdown("#### 🔍 Teks Plat Terdeteksi:")
    st.success(f"`{extracted_text}`")

    # Cari angka 4 digit sebagai tahun
    import re
    year_match = re.findall(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", extracted_text)
    if year_match:
        ocr_year = int(year_match[0])
        st.info(f"Tahun Produksi terdeteksi dari plat: **{ocr_year}**")

# ============== INPUT USER ============== #
with st.container():
    car_models = sorted(list(set(pd.read_csv('toyota.csv')['model'].unique())))
    selected_model = st.selectbox('Model Mobil', car_models)

    transmissions = ['Manual', 'Automatic', 'Semi-Auto']
    selected_transmission = st.selectbox('Transmisi', transmissions)

    fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Other']
    selected_fuel_type = st.selectbox('Jenis Bahan Bakar', fuel_types)

    col1, col2 = st.columns(2)
    with col1:
        # Gunakan hasil OCR jika tersedia
        year = st.number_input('Tahun Produksi', min_value=2001, max_value=2024,
                               step=1, value=ocr_year if ocr_year else 2020)
    with col2:
        mileage = st.number_input('Jarak Tempuh (KM)', min_value=0)

# ============== FORMAT CURRENCY ============== #
def format_price(number):
    return "{:,.0f}".format(number).replace(",", ".")

# ============== PREDIKSI ============== #
if st.button('🔮 Prediksi Harga'):
    if year == 0 or mileage == 0:
        st.warning('Mohon lengkapi semua data input!')
    else:
        with st.spinner('Memproses prediksi...'):
            try:
                # Prepare categorical features
                categorical_features = pd.DataFrame({
                    'model': [selected_model],
                    'transmission': [selected_transmission],
                    'fuelType': [selected_fuel_type]
                })
                encoded_categorical = encoder.transform(categorical_features)
                
                # Numerical
                numerical_features = np.array([[year, mileage]])
                
                # Combine features
                prediction_input = np.hstack((numerical_features, encoded_categorical))
                
                # Predict
                prediction = model.predict(prediction_input)[0]
                prediction_rupiah = prediction * 19500
                
                st.success('✅ Prediksi Selesai!')
                st.markdown(f"### 💰 Prediksi Harga Mobil: `Rp {format_price(prediction_rupiah)}`")
                
                st.markdown("---")
                st.markdown(f"- 📊 MAE: **{metrics['mae']:.2f}**")
                st.markdown(f"- 📉 MAPE: **{metrics['mape']:.2f}%**")
                st.markdown(f"- 🧠 Akurasi Model: **{metrics['accuracy']:.2f}%**")
            except Exception as e:
                st.error(f'❌ Terjadi kesalahan: {str(e)}')
