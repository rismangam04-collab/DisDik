# app_streamlit.py
# Dashboard AI Prediksi Dropout & Rekomendasi Jalur Pendidikan
# Install dulu: pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl

from importlib.metadata import version 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Dashboard AI Pendidikan", layout="wide")

# ---------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------
def preprocess(df):
    required = ["nama","tgl_lahir","tgl_masuk_sd","kelas_terakhir","status",
                "tunggakan","pendapatan_keluarga","alasan_putus","sekolah_asal_tipe","alamat_kecamatan"]
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    df["tgl_lahir"] = pd.to_datetime(df["tgl_lahir"], errors="coerce")
    df["tgl_masuk_sd"] = pd.to_datetime(df["tgl_masuk_sd"], errors="coerce")
    df["usia"] = ((pd.Timestamp.now() - df["tgl_lahir"]).dt.days / 365).round(1)
    df["usia_masuk_sd"] = ((df["tgl_masuk_sd"] - df["tgl_lahir"]).dt.days / 365).round(1)

    df["tunggakan"] = pd.to_numeric(df["tunggakan"], errors="coerce").fillna(0)
    df["pendapatan_keluarga"] = pd.to_numeric(df["pendapatan_keluarga"], errors="coerce").fillna(0)

    df["kategori_tunggakan"] = pd.cut(df["tunggakan"],
                                      bins=[-1, 0, 500000, 2000000, np.inf],
                                      labels=["Lunas","Ringan","Sedang","Tinggi"])

    df["alasan_putus_norm"] = df["alasan_putus"].fillna("").astype(str).str.strip().str.lower()

    def map_alasan(txt):
        if "ekonomi" in txt or "uang" in txt or "biaya" in txt:
            return "Ekonomi"
        elif "pindah" in txt:
            return "Pindah Orang Tua"
        elif "minat" in txt or "malas" in txt or "bosan" in txt:
            return "Minat Rendah"
        elif "kerja" in txt:
            return "Bekerja"
        elif "sakit" in txt or "kesehatan" in txt:
            return "Kesehatan"
        elif "pergaulan" in txt or "narkoba" in txt:
            return "Pergaulan Buruk"
        elif "jarak" in txt or "transport" in txt:
            return "Jarak Sekolah"
        else:
            return "Lainnya"
    df["alasan_kategori"] = df["alasan_putus_norm"].apply(map_alasan)
    df["label_dropout"] = np.where(df["status"].astype(str).str.lower()=="putus", 1, 0)
    return df

# ---------------------------------------------------
# FUNGSI REKOMENDASI JALUR PENDIDIKAN - VERSI FINAL TERBAIK
# ---------------------------------------------------
def rekomendasi_penempatan(row):
    usia = row["usia"]
    kelas = row["kelas_terakhir"]
    status = str(row["status"]).lower()
    alasan = row["alasan_kategori"]

    # Jika data penting kosong
    if pd.isna(usia) or pd.isna(kelas):
        return "⚠️ Data belum lengkap, mohon lengkapi tanggal lahir & kelas terakhir"

    try:
        kelas = int(kelas)
    except:
        return "⚠️ Data kelas tidak valid, perbaiki data siswa"

    # Batas usia ideal per kelas SD (Kemendikbud)
    batas_usia_kelas = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12}

    # =======================
    # 1. SISWA MASIH AKTIF
    # =======================
    if status == "aktif":
        if kelas in batas_usia_kelas:
            usia_normal = batas_usia_kelas[kelas]
            if usia > usia_normal + 2:
                return "📌 Direkomendasikan Paket A (Usia melebihi batas wajar untuk SD)"
            else:
                return "✅ Tetap di Sekolah Reguler"
        return "⚠️ Kelas tidak valid, periksa data siswa"

    # =======================
    # 2. SISWA SUDAH LULUS SD
    # =======================
    if status == "lulus":
        if kelas == 6:
            if usia <= 15:
                return "🎓 Lanjut ke SMP Reguler"
            else:
                return "📌 Direkomendasikan Paket B (Setara SMP, karena usia di atas standar)"
        return "⚠️ Data lulus tidak konsisten, periksa kelas terakhir"

    # =======================
    # 3. SISWA PUTUS SEKOLAH
    # =======================
    if status == "putus":
        # Jika usia masih cukup muda untuk lanjut sekolah reguler
        if usia <= 12 and kelas <= 5:
            return "🎯 Rekomendasi Sekolah Terdekat (Masih cukup usia untuk lanjut SD)"

        # Jika sudah tamat SD tapi putus saat mau lanjut SMP
        if kelas == 6 and usia <= 15:
            return "🎯 Rekomendasi Lanjut SMP Reguler"

        # Putus karena alasan ekonomi → tawarkan Paket A
        if alasan == "Ekonomi":
            return "📌 Direkomendasikan Paket A (Gratis, setara SD)"

        # Putus karena pindah orang tua
        elif alasan == "Pindah Orang Tua":
            return "🏫 Daftar ke Sekolah Terdekat sesuai domisili baru"

        # Putus karena minat rendah / pergaulan buruk
        elif alasan in ["Minat Rendah", "Pergaulan Buruk"]:
            return "👨‍🏫 Program Bimbingan Konseling & Kejar Paket A"

        # Putus karena jarak sekolah terlalu jauh
        elif alasan == "Jarak Sekolah":
            return "🏫 Direkomendasikan Sekolah Terdekat"

        # Putus karena kesehatan
        elif alasan == "Kesehatan":
            return "🏥 Sekolah Inklusi / Program Pendidikan Khusus"

        # Jika kelas 2-6 tetapi usia ≥ 13 tahun → Paket A
        if kelas in [2, 3, 4, 5, 6] and usia >= 13:
            return "📌 Direkomendasikan Paket A (Karena usia 13 tahun ke atas)"

        # Jika usia sangat tinggi (≥15) dan belum tamat SD → Paket A
        if usia >= 15 and kelas <= 6:
            return "📌 Direkomendasikan Paket A (Usia jauh di atas standar SD)"

        # Jika putus setelah lulus SD dan usia ≥15 → Paket B
        if kelas == 6 and usia >= 15:
            return "📌 Direkomendasikan Paket B (Setara SMP)"

        # Jika tidak memenuhi kondisi lain
        return "⚠️ Jalur pendidikan tidak terdefinisi, perlu cek data siswa"

    # =======================
    # 4. STATUS TIDAK JELAS
    # =======================
    return "⚠️ Status siswa tidak jelas, periksa kembali data"


# ---------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------
st.title("📊 Dashboard AI Pendidikan - Prediksi Dropout & Jalur Penempatan")
st.markdown("""
### Upload CSV dengan format berikut:
**nama, tgl_lahir, tgl_masuk_sd, kelas_terakhir, status, tunggakan, pendapatan_keluarga, alasan_putus, sekolah_asal_tipe, alamat_kecamatan**
""")

uploaded = st.file_uploader("Upload data_siswa.csv", type=["csv"])
sample_btn = st.button("Gunakan Data Contoh")
df = None

if uploaded:
    df = pd.read_csv(uploaded)
elif sample_btn and os.path.exists("data_siswa.csv"):
    df = pd.read_csv("data_siswa.csv")

if df is not None:
    dfp = preprocess(df.copy())
    dfp["rekomendasi_penempatan"] = dfp.apply(rekomendasi_penempatan, axis=1)

    st.subheader("🔍 Preview Data")
    st.dataframe(dfp.head(10))

    # Statistik
    st.subheader("📈 Statistik Umum")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Siswa", len(dfp))
    with col2:
        st.metric("Total Putus", int(dfp["label_dropout"].sum()))
    with col3:
        st.metric("Total Tunggakan", f"Rp {int(dfp['tunggakan'].sum()):,}")

    # Visualisasi: Pie Chart Status
    st.subheader("🟢 Distribusi Status Siswa")
    fig1, ax1 = plt.subplots()
    dfp["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1, cmap="coolwarm")
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # Visualisasi: Bar Chart Jalur Rekomendasi
    st.subheader("📌 Distribusi Jalur Pendidikan Rekomendasi")
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.countplot(x="rekomendasi_penempatan", data=dfp, palette="Set2", ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

    # Data rekomendasi
    st.subheader("📌 Rekomendasi Penempatan Detail")
    st.dataframe(dfp[["nama","usia","kelas_terakhir","status","alasan_kategori","rekomendasi_penempatan"]])

    # Download hasil rekomendasi
    csv = dfp.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Rekomendasi", data=csv, file_name="rekomendasi_siswa.csv", mime="text/csv")
