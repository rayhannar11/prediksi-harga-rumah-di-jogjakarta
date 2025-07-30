import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan data
model = pickle.load(open('best_rf_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))
data = pd.read_csv('rumah123_yogya_clean.csv')

st.title("Prediksi Harga Properti di Yogyakarta")

# --- Tombol Reset ---
if st.button("Reset"):
    for key in st.session_state.keys():
        st.session_state[key] = ''
    st.rerun()

# --- Inisialisasi nilai kosong di session_state ---
fields = ['bed', 'bath', 'carport', 'surface_area', 'building_area', 'kecamatan', 'kabupaten']
for field in fields:
    if field not in st.session_state:
        st.session_state[field] = ''

# --- Input teks agar bisa kosong ---
bed = st.text_input("Jumlah Kamar Tidur", value=st.session_state['bed'], key='bed')
bath = st.text_input("Jumlah Kamar Mandi", value=st.session_state['bath'], key='bath')
carport = st.text_input("Jumlah Carport", value=st.session_state['carport'], key='carport')
surface_area = st.text_input("Luas Tanah (m²)", value=st.session_state['surface_area'], key='surface_area')
building_area = st.text_input("Luas Bangunan (m²)", value=st.session_state['building_area'], key='building_area')

# --- Dropdown Lokasi ---
kecamatan_list = sorted(data['kecamatan'].dropna().unique().tolist())

if st.session_state['kecamatan'] in kecamatan_list:
    idx_kec = kecamatan_list.index(st.session_state['kecamatan']) + 1
else:
    idx_kec = 0

kecamatan = st.selectbox("Pilih Kecamatan", [''] + kecamatan_list, index=idx_kec, key='kecamatan')

# pilih kabupaten tergantung kecamatan
if st.session_state['kecamatan']:
    kabupaten_list = sorted(data[data['kecamatan'] == st.session_state['kecamatan']]['kabupaten'].dropna().unique().tolist())
else:
    kabupaten_list = []

if st.session_state['kabupaten'] in kabupaten_list:
    idx_kab = kabupaten_list.index(st.session_state['kabupaten']) + 1
else:
    idx_kab = 0

kabupaten = st.selectbox("Pilih Kabupaten/Kota", [''] + kabupaten_list, index=idx_kab, key='kabupaten')

# --- Prediksi harga ---
if st.button("Prediksi Harga"):
    try:
        # Konversi input ke tipe data numerik
        bed = int(st.session_state['bed'])
        bath = int(st.session_state['bath'])
        carport = int(st.session_state['carport'])
        surface_area = float(st.session_state['surface_area'])
        building_area = float(st.session_state['building_area'])

        # Validasi nilai tidak boleh 0 atau kosong
        if any([
            bed <= 0,
            bath <= 0,
            carport <= 0,
            surface_area <= 0,
            building_area <= 0,
            not st.session_state['kecamatan'],
            not st.session_state['kabupaten']
        ]):
            st.warning("Semua input harus diisi dan bernilai lebih dari 0.")
        else:
            lokasi_full = f"{st.session_state['kecamatan']}, {st.session_state['kabupaten']}"
            if lokasi_full in le.classes_:
                loc_encoded = le.transform([lokasi_full])[0]
                price_per_m2 = data[data['loc_encoded'] == loc_encoded]['price_per_m2'].median()
            else:
                loc_encoded = 0
                price_per_m2 = data['price_per_m2'].median()
                st.info("Lokasi tidak ditemukan, model menggunakan nilai rata-rata semua lokasi.")

            input_data = pd.DataFrame([{
                'bed': bed,
                'bath': bath,
                'carport': carport,
                'surface_area': surface_area,
                'building_area': building_area,
                'price_per_m2': price_per_m2,
                'loc_encoded': loc_encoded
            }])

            prediksi = model.predict(input_data)[0]
            st.success(f"Prediksi Harga Properti: Rp {prediksi:,.2f}")

    except ValueError:
        st.error("Mohon isi semua input angka dengan benar (tidak boleh kosong atau huruf).")
