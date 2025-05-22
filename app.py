# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model dan fitur
model = joblib.load("model_lung_cancer.pkl")
feature_order = joblib.load("feature_columns.pkl")

# Judul Aplikasi
st.title("🫁Prediksi Penyakit Kanker Paru-paru")
st.markdown("Silakan isi data berikut untuk melakukan prediksi kemungkinan risiko kanker paru-paru.")

# Fungsi bantu untuk konversi nilai
def encode_binary(val):
    return 1 if val == "Ya" else 0

# Input Usia
AGE = st.number_input("Usia (dalam tahun)", min_value=20, max_value=100, value=50)

# Daftar pertanyaan biner
st.markdown("### 📝 Kuesioner Gejala & Riwayat")
questions = {
    "GENDER": "Apakah Anda seorang laki-laki?",
    "SMOKING": "Apakah Anda merokok? (Baik rokok aktif maupun vape)",
    "FINGER_DISCOLORATION": "Apakah ujung jari Anda terlihat kekuningan atau berubah warna secara tidak normal?",
    "EXPOSURE_TO_POLLUTION": "Apakah Anda sering terpapar polusi udara? (Contoh: asap kendaraan, asap rokok, lingkungan kerja berdebu)",
    "LONG_TERM_ILLNESS": "Apakah Anda memiliki riwayat penyakit kronis atau jangka panjang? (Contoh: asma, diabetes, hipertensi, TBC, dll.)",
    "IMMUNE_WEAKNESS": "Apakah Anda merasa memiliki daya tahan tubuh yang lemah atau mudah sakit?",
    "BREATHING_ISSUE": "Apakah Anda sering mengalami masalah pernapasan seperti sesak atau napas pendek?",
    "ALCOHOL_CONSUMPTION": "Apakah Anda mengonsumsi minuman beralkohol secara rutin atau pernah?",
    "THROAT_DISCOMFORT": "Apakah Anda sering merasa tidak nyaman di tenggorokan, seperti sakit saat menelan sesuatu?",
    "CHEST_TIGHTNESS": "Apakah Anda pernah merasakan dada terasa sesak, berat, atau nyeri?",
    "FAMILY_HISTORY": "Apakah ada anggota keluarga Anda yang pernah menderita kanker paru-paru?",
    "SMOKING_FAMILY_HISTORY": "Apakah ada anggota keluarga Anda yang merokok secara rutin di rumah?",
    "STRESS_IMMUNE": "Apakah stres berdampak pada daya tahan tubuh Anda? (Contoh: mudah sakit saat stres)",
    "MENTAL_STRESS": "Apakah stres atau tekanan pikiran sering memengaruhi kondisi fisik Anda?",
    "ENERGY_LEVEL_KATEGORI": "Apakah Anda merasa tubuh Anda mudah lelah tanpa alasan dalam aktivitas sehari-hari?",
    "OXYGEN_SATURATION_KATEGORI": "Apakah kadar saturasi oksigen Anda pernah di bawah 95% atau sering merasa kekurangan oksigen?"
}

# Tempat menyimpan input pengguna
user_input = {}

# Input untuk setiap pertanyaan biner
for key, question in questions.items():
    response = st.radio(question, ["-", "Tidak", "Ya"], key=key)
    if response == "-":
        st.warning(f"Silakan jawab pertanyaan ")
        st.stop()
    user_input[key] = encode_binary(response)

# Gabungkan dengan input usia
input_data = pd.DataFrame([{
    'AGE': AGE,
    **user_input
}])

# Reorder kolom sesuai dengan saat pelatihan model
input_data = input_data[feature_order]

# Tombol untuk prediksi
if st.button("🔍 Lakukan Prediksi"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    confidence = probabilities[prediction]

    # Hasil
    if prediction == 1:
        st.subheader("Hasil Prediksi:")
        st.error("POSITIF – Berdasarkan hasil prediksi, Anda **berpotensi memiliki gejala atau risiko kanker paru-paru**. "
                 "Disarankan untuk segera berkonsultasi dengan tenaga medis profesional untuk pemeriksaan lebih lanjut.")
    else:
        st.subheader("📊 Hasil Prediksi:")
        st.success("NEGATIF – Berdasarkan hasil prediksi, Anda **tidak menunjukkan indikasi signifikan terhadap kanker paru-paru**. "
                   "Tetap jaga kesehatan dan lakukan pemeriksaan rutin bila diperlukan.")

    st.info(f"🔍 Tingkat Keyakinan Model: **{confidence * 100:.2f}%**")
