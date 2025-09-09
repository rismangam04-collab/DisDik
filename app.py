# app_streamlit.py
# Dashboard AI Prediksi Dropout & Rekomendasi Jalur Pendidikan
# Install package: pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------
st.set_page_config(page_title="Dashboard AI Pendidikan", layout="wide")
st.title("📊 Dashboard AI Pendidikan - Prediksi Dropout & Jalur Penempatan")

# ---------------------------------------------------
# INFORMASI KOMPONEN DATA YANG HARUS ADA
# ---------------------------------------------------
("Pastikan file **data.csv** memiliki kolom berikut: NO , RT,RW, Kecamatan, Nama Posyandu, Kelurahan, Nama Anak Lenkap Sesuai Dokumen, Jenis kelamin Anak, Tempat Lahir Anak, Tanggal Lahir Anak, Nama Orang tua/wali, Alamat Lengap, Keterangan Tambahan, Nomor hp/WA, Alasan Tidak Sekolah, Pendidikan Terakhir(harus ada angkanya contoh 2), Status Domisili, Keterangan Tambahan (Misalnya: Sudah Bekerja,Status Siswa(putus, aktif, lulus), Tunggakan, Ingin Sekolah Lagi, Sedang Ikut Paket A/B/C, dll). ")

("unutk mengambil haslinya  copy saja datanya  paste"
" ke excel, jika ada data yang kosong masi tetap  bisa di proses. sebelum itu utk mengexstrak data nya jika excel uah dulu ke csv(comma delimited).")  
   

# ---------------------------------------------------
# FUNGSI BACA CSV DENGAN PERLINDUNGAN
# ---------------------------------------------------
def load_csv(uploaded):
    try:
        return pd.read_csv(uploaded, encoding="utf-8", sep=",", quotechar='"', on_bad_lines="skip")
    except:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="latin1", sep=";", quotechar='"', on_bad_lines="skip")

# ---------------------------------------------------
# PREPROCESS DATA POSYANDU (REVISI)
# ---------------------------------------------------
def preprocess(df):
    # Mapping nama kolom
    mapping = {
        "Nama Anak Lengkap Sesuai Dokumen": "nama",
        "Tanggal Lahir Anak": "tgl_lahir",
        "Pendidikan Terakhir": "kelas_terakhir",
        "Alasan Tidak Sekolah": "alasan_putus",
        "Kecamatan": "alamat_kecamatan",
        "Tunggakan": "tunggakan",
        "Status Siswa": "status"  # pastikan jika ada kolom ini, langsung dipakai
    }
    df = df.rename(columns=mapping)

    # ---- Konversi kolom tunggakan ke angka
    if "tunggakan" not in df.columns:
        df["tunggakan"] = 0
    else:
        df["tunggakan"] = (
            df["tunggakan"]
            .astype(str)
            .str.replace(r"[^\d]", "", regex=True)
            .replace("", "0")
            .astype(float)
        )

    # ---- Pastikan status terbaca dengan benar
    if "status" in df.columns:
        # Jika ada kolom status, pakai langsung & normalisasi
        df["status"] = (
            df["status"]
            .fillna("aktif")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        # Normalisasi supaya seragam
        df["status"] = df["status"].replace({
            "putus sekolah": "putus",
            "drop out": "putus",
            "berhenti": "putus",
            "aktif sekolah": "aktif",
            "masih sekolah": "aktif",
            "lulus sekolah": "lulus",
            "kelulusan": "lulus"
        })
    else:
        # Kalau kolom status tidak ada → tentukan otomatis
        def tentukan_status(row):
            alasan = str(row["alasan_putus"]).strip().lower()
            if alasan == "" or alasan == "nan":
                return "aktif"
            elif "lulus" in alasan:
                return "lulus"
            elif "putus" in alasan or "masalah" in alasan:
                return "putus"
            return "aktif"

        df["status"] = df.apply(tentukan_status, axis=1)

    # ---- Hitung usia dari tanggal lahir
    df["tgl_lahir"] = pd.to_datetime(df["tgl_lahir"], errors="coerce")
    df["usia"] = ((pd.Timestamp.now() - df["tgl_lahir"]).dt.days / 365).round(1)

    # ---- Konversi kelas terakhir → angka
    def convert_kelas(kelas):
        if pd.isna(kelas):
            return np.nan
        kelas = str(kelas).strip().upper()
        for i in range(1, 13):
            if str(i) in kelas:
                return i
        if "XII" in kelas: return 12
        if "XI" in kelas: return 11
        if "X" in kelas: return 10
        return np.nan

    df["kelas_terakhir"] = df["kelas_terakhir"].apply(convert_kelas)

    # ---- Normalisasi alasan putus
    df["alasan_putus_norm"] = df["alasan_putus"].fillna("").astype(str).str.lower()

    def map_alasan(txt):
        if "ekonomi" in txt or "uang" in txt or "biaya" in txt:
            return "Ekonomi"
        elif "ijazah" in txt or "tunggakan" in txt:
            return "Ijazah/Tunggakan"
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

    # ---- Label dropout untuk statistik
    df["label_dropout"] = np.where(df["status"] == "putus", 1, 0)
    return df

# ---------------------------------------------------
# FUNGSI REKOMENDASI JALUR PENDIDIKAN (REVISI)
# ---------------------------------------------------
def rekomendasi_penempatan(row):
    usia = row.get("usia", np.nan)
    kelas = row.get("kelas_terakhir", np.nan)
    status = str(row.get("status", "")).lower()
    alasan = row.get("alasan_kategori", "")
    tunggakan = row.get("tunggakan")

    if pd.isna(usia) and pd.isna(kelas):
        return "⚠️ Data kurang lengkap"

    try:
        kelas = int(kelas)
    except:
        kelas = np.nan

    batas_usia_kelas = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12}

    # STATUS AKTIF → lanjut sekolah
    if status == "aktif":
        if not pd.isna(kelas):
            usia_normal = batas_usia_kelas.get(kelas, 12)
            if usia > usia_normal + 2:
                return "📌 Direkomendasikan Paket A (Usia lebih dari standar SD)"
            return "✅ Tetap di Sekolah Reguler"
        return "✅ Tetap sekolah, kelas tidak terdeteksi"

    # STATUS LULUS → lanjut ke jenjang berikutnya
    if status == "lulus":
        if kelas == 6:
            return "🎓 Lanjut ke SMP Reguler"
        elif kelas >= 9:
            return "🎓 Lanjut ke SMA/SMK"
        return "⚠️ Perlu cek data kelulusan"

    # STATUS PUTUS SEKOLAH
    if status == "putus":
        # Kasus EKONOMI → pertimbangkan usia + kelas + tunggakan
        if alasan == "Ekonomi":
            if kelas <= 5 and usia <= 12:
                return "🎯 Lanjut SD Reguler (Bantuan Program Ekonomi)"
            if kelas == 6:
                if usia <= 15:
                    if tunggakan > 0:
                        return "🎯 Lanjut SMP Reguler (Setelah melunasi tunggakan)"
                    else:
                        return "🎯 Lanjut SMP Reguler (Layak tanpa tunggakan)"
                else:
                    return "📌 Paket B (Setara SMP, usia di atas standar)"
            if kelas <= 5 and usia >= 13:
                return "📌 Paket A (Setara SD, usia di atas standar)"
            return "👨‍🏫 Program Kejar Paket A/B sesuai kondisi"

        # Kasus IJAZAH/TUNGGAKAN → boleh lanjut SMP jika syaratnya terpenuhi
        if alasan == "Ijazah/Tunggakan":
            if kelas == 6 and usia <= 15:
                return "🎯 Lanjut SMP Reguler (Setelah melunasi tunggakan/ijazah)"
            elif kelas == 6 and usia > 15:
                return "📌 Paket B (Setara SMP, usia di atas standar)"
            else:
                return "📌 Paket A (Setara SD)"

        # Kasus lainnya → logika default
        if kelas <= 5 and usia <= 12:
            return "🎯 Lanjut SD Reguler"
        if kelas <= 6 and usia >= 13:
            return "📌 Paket A (Usia di atas standar SD)"
        if kelas == 6 and usia >= 15:
            return "📌 Paket B (Setara SMP)"
        return "👨‍🏫 Program Kejar Paket A/B sesuai usia"

    return "⚠️ Status siswa tidak jelas"

# ---------------------------------------------------
# STREAMLIT DASHBOARD
# ---------------------------------------------------
uploaded = st.file_uploader("📂 Upload data.csv", type=["csv"])
sample_btn = st.button("Gunakan Data Contoh")
df = None

if uploaded:
    df = load_csv(uploaded)
elif sample_btn and os.path.exists("data_posyandu.csv"):
    df = load_csv("data_posyandu.csv")

if df is not None:
    dfp = preprocess(df.copy())
    dfp["rekomendasi_penempatan"] = dfp.apply(rekomendasi_penempatan, axis=1)

    # Statistik umum
    st.subheader("📈 Statistik Umum Siswa")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Siswa", len(dfp))
    with col2:
        st.metric("Total Putus Sekolah", int(dfp["label_dropout"].sum()))
    with col3:
        st.metric("Total Tunggakan", f"Rp {int(dfp['tunggakan'].sum()):,}")

    # Distribusi status
    st.subheader("🟢 Distribusi Status Siswa")
    fig1, ax1 = plt.subplots()
    dfp["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1, cmap="coolwarm")
    ax1.set_ylabel("")
    st.pyplot(fig1)

    # Distribusi rekomendasi
    st.subheader("📌 Distribusi Jalur Pendidikan Rekomendasi")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.countplot(x="rekomendasi_penempatan", data=dfp, palette="Set2", ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    st.pyplot(fig2)

    # Tabel detail rekomendasi
    st.subheader("📌 Rekomendasi Penempatan Detail")
    st.dataframe(dfp[["nama", "usia", "kelas_terakhir", "status", "alasan_kategori", "tunggakan", "rekomendasi_penempatan"]],
                 use_container_width=True, height=600)

    # Tombol download hasil rekomendasi
    csv = dfp.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Hasil Rekomendasi", data=csv, file_name="rekomendasi_siswa.csv", mime="text/csv")
