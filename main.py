import math
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Prediksi Produksi Padi Menggunakan Metode <br> <i>Support Vector Regression<i/> (SVR) </h1>", unsafe_allow_html=True
)


with st.sidebar:
    selected = option_menu("Main Menu", ['Dataset', 'Preprocessing', 'Modelling', 'Prediction'], default_index=0)

# memuat dataset
data = pd.read_excel('data.xlsx')

# variabel yang digunakan
var = data.drop(columns=['Kabupaten', 'Tahun'], axis =1)
var_copy = data.drop(columns=['Kabupaten', 'Tahun'], axis =1)

# label encoder
label_encoder = LabelEncoder()
var['Kecamatan'] = label_encoder.fit_transform(var['Kecamatan'])
variabel = var

# missing values
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(variabel), columns=variabel.columns)

# penanganan outlier
pusat_data = data_imputed.mean().values
jarak = data_imputed.apply(lambda row: euclidean(row.values, pusat_data), axis=1)
z = np.abs(stats.zscore(jarak))

threshold_z = 3
identifikasi_outlier = np.where(z > threshold_z)[0]
outliers = np.unique(identifikasi_outlier)
data_bersih = data_imputed.drop(outliers)

# Mask untuk nilai individual yang outlier
outlier_mask = z > threshold_z
data_outlier_values = data_imputed[outlier_mask]

# Ubah ke format long untuk detail
outlier_details = data_outlier_values.stack().reset_index()
outlier_details.columns = ['Row', 'Column', 'Value']


# fitur dan target
data_prepro = data_bersih.values
fitur = data_prepro[:, :-1]
target = data_prepro[:, -1].reshape(-1, 1)

# normalisasi data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(fitur)
y_scaled = scaler_y.fit_transform(target)

# pemodelan
# ======== Split data ========
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

with open('best_svr_model.pkl', 'rb') as f:
    loaded = pickle.load(f)

best_model = loaded['model']
best_params = loaded['best_params']
results_df = pd.DataFrame(loaded['results'])
best_model.fit(X_train, y_train.ravel())


if (selected == 'Dataset'):
    st.image("image.jpg", caption="Ilustrasi menanam padi", use_column_width=True)
    st.info("TUJUAN PENELITIAN")
    st.write("""
Penelitian ini dilakukan dengan tujuan untuk menganalisis pengaruh penggunaan kernel dan parameter dari metode Support Vector Regression (SVR)  dalam membangun model prediksi produksi padi serta menilai seberapa baik model yang dibangun dalam meminimalkan nilai Mean Absolute Percentage Error (MAPE).
    """)

    st.info("DATA PENELITIAN")
    st.write("Data yang digunakan dalam penelitian ini diperoleh dari Dinas Pertanian Kabupaten Bangkalan, Sampang, dan Pamekasan. Data tersebut mencakup informasi mengenai luas tanam dan luas panen di setiap kecamatan selama periode 2018 hingga 2023. Sementara itu, data curah hujan untuk periode yang sama diperoleh melalui publikasi resmi dari Badan Pusat Statistik (BPS).")
    col1, col2 = st.columns(2)
    with col1:
        data
    with col2:
        st.write('Keterangan')
        st.write("""
            1. Kabupaten   : Nama kabupaten dari tempat pengambilan data
            2. Kecamatan   : Nama kecamatan dari tempat pengambilan data
            3. Luas Tanam  : Luas lahan yang digunakan untuk menanam padi dalam satuan hektar (Ha)
            4. Luas Panen  : Luas lahan yang berhasil dipanen dari total luas tanam dalam satuan hektar (Ha)
            5. Curah Hujan : Volume curah hujan yang turun di suatu daerah dalam jangka waktu satu tahun dalam satuan milimeter (mm)
            6. Produksi    : Total hasil padi yang dipanen dari seluruh luas panen dalam satuan ton (ton)
        """)

if (selected == 'Preprocessing'):
    st.info("""
    Adapun tahapan - tahapan yang akan dilakukan pada preprocessing ini adalah :
    1. Tahap konversi data menggunakan Label Encoder
    2. Tahap imputasi data menggunakan Imputasi KNN
    3. Tahap penanganan outler menggunakan Z-score
    4. Tahap normalisasi data menggunakan min-max scalling
    """)

    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["Memuat Dataset", "Transformasi Data", "Imputasi Data", "Penanganan Outlier", "Normalisasi Data", "Dataset"])

    with tab1:
        st.warning('MEMUAT DATASET')
        st.code("""
import pandas as pd
data = pd.read_excel('data.xlsx')
""")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data')
            data
            st.write(f'Total data yang digunakan : { data.shape[0] } data')
        with col2:
            st.subheader('Informasi Data')
            jumlah_kecamatan = data.groupby('Kabupaten')['Kecamatan'].nunique()
            jumlah_kecamatan.columns = ['Kabupaten', 'Jumlah Kecamatan']
            st.write(jumlah_kecamatan)
            st.info("Variabel Yang Digunakan")
            st.write(var_copy.dtypes)

        st.title("Visualisasi Data Pertanian per Kecamatan")

        # Bersihkan nama kolom
        data.columns = [col.strip() for col in data.columns]

        # Ambil daftar kecamatan unik
        kec_list = data['Kecamatan'].unique()

        # Buat layout per kecamatan (gunakan 3 kolom per baris)
        cols = 5
        rows = math.ceil(len(kec_list) / cols)

        for i in range(rows):
            col_row = st.columns(cols)
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(kec_list):
                    break

                kec = kec_list[idx]
                df_kec = data[data['Kecamatan'] == kec].sort_values('Tahun')

                fig, ax = plt.subplots(figsize=(6, 4))

                # Grafik garis fitur
                ax.plot(df_kec['Tahun'], df_kec['Luas Tanam (Ha)'], label='Luas Tanam', marker='o', color='tab:green')
                ax.plot(df_kec['Tahun'], df_kec['Luas Panen (Ha)'], label='Luas Panen', marker='o', color='tab:red')
                ax.plot(df_kec['Tahun'], df_kec['Curah Hujan (mm)'], label='Curah Hujan', marker='o', color='tab:blue')

                # Twin axis untuk produksi
                ax2 = ax.twinx()
                ax2.bar(df_kec['Tahun'], df_kec['Produksi (Ton)'], alpha=0.3, color='tab:orange', label='Produksi')

                # Judul dan label
                ax.set_title(f'{kec}', fontsize=10)
                ax.set_xlabel('Tahun')
                ax.set_ylabel('Nilai Fitur')
                ax2.set_ylabel('Produksi', color='tab:orange')

                # Gabungkan legenda
                lines, labels = ax.get_legend_handles_labels()
                bars, bar_labels = ax2.get_legend_handles_labels()
                ax2.legend(lines + bars, labels + bar_labels, loc='upper left', fontsize=8)

                # Tampilkan chart di kolom streamlit
                col_row[j].pyplot(fig)
      
    with tab2:
        st.warning('KONVERSI DATA | Dilakukan untuk mengubah data kategorial menjadi numerik agar bisa diproses dan dianalisi oleh komputer menggunakan label encoder.')
        st.code("""
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
data['Kecamatan'] = label_encoder.fit_transform(data['Kecamatan'])
""")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader('Data Sebelum')
            var_copy
        with col2:
            st.subheader('Data Setelah')
            variabel
        with col3:
            st.write("Keterangan")
            mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
            mapping

    with tab3:
        st.warning('IMPUTASI DATA | Dilakukan untuk menangani data yang hilang menggunakan imputasi KNN  ')
        st.code("""
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
""")
        st.write("Jumlah Missing Values dalam Setiap Kolom : ", variabel.isnull().sum())
        
        st.subheader('Data Sebelum Imputasi Data')
        col1, col2 = st.columns(2)
        with col1:
            variabel
        with col2:
            data_grafik = variabel.copy()
            data_grafik['Kecamatan'] = label_encoder.inverse_transform(data_grafik['Kecamatan'].astype(int))
            data_grafik['ID'] = data_grafik.groupby('Kecamatan').cumcount()
            data_grafik['Label'] = data_grafik['Kecamatan'] + ' - ' + data_grafik['ID'].astype(str)

            st.subheader("Visualisasi Data Curah Hujan per Kecamatan")
            fig, ax = plt.subplots(figsize=(20, 6))
            sns.barplot(data=data_grafik, x='Label', y='Curah Hujan (mm)', palette='viridis', ax=ax)
            plt.xticks(rotation=90)
            plt.title('Curah Hujan per Kecamatan')
            plt.xlabel('Kecamatan (Setiap Baris Data)')
            plt.ylabel('Curah Hujan (mm)')
            st.pyplot(fig)


        st.subheader('Data Setelah Imputasi Data')
        col3, col4 = st.columns(2)
        with col3:
            data_imputed
        with col4:
            data_grafik = data_imputed.copy()
            data_grafik['Kecamatan'] = label_encoder.inverse_transform(data_grafik['Kecamatan'].astype(int))
            data_grafik['ID'] = data_grafik.groupby('Kecamatan').cumcount()
            data_grafik['Label'] = data_grafik['Kecamatan'] + ' - ' + data_grafik['ID'].astype(str)

            st.subheader("Visualisasi Data Curah Hujan per Kecamatan")
            fig, ax = plt.subplots(figsize=(20, 6))
            sns.barplot(data=data_grafik, x='Label', y='Curah Hujan (mm)', palette='viridis', ax=ax)
            plt.xticks(rotation=90)
            plt.title('Curah Hujan per Kecamatan')
            plt.xlabel('Kecamatan (Setiap Baris Data)')
            plt.ylabel('Curah Hujan (mm)')
            st.pyplot(fig)
        missing_values = data_imputed.isnull().sum()
        st.write("Jumlah Missing Values dalam Setiap Kolom Setelah Imputasi Data : ", missing_values)


    with tab4:
        st.warning('PENANGANAN OUTLIER| Dilakukan untuk menghilangkan nilai ekstrem yang dapat memengaruhi hasil analisis menggunakan Z-Score')
        st.code("""
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
                
pusat_data = data_imputed.mean().values
jarak = data_imputed.apply(lambda row: euclidean(row.values, pusat_data), axis=1)
z = np.abs(stats.zscore(jarak))
threshold_z = 3

identifikasi_outlier = np.where(z > threshold_z)[0]
outliers = np.unique(identifikasi_outlier)
data_bersih = data_imputed.drop(outliers)
""")
        st.write("Berikut data outlier yang ditemukan : ")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baris yang termasuk outlier:")
            data_outlier_rows = data_imputed.iloc[outliers]
            st.dataframe(data_outlier_rows)

        with col2:
            st.error(f"Ditemukan outlier sebanyak {len(outliers)} baris data")
        
        st.subheader('Sebelum Penangan Outlier')
        col1, col2 = st.columns(2)
        with col1:
            data_imputed
        with col2:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=data_imputed, palette="tab10", dashes=False, ax=ax2)
            ax2.set_title("Sebelum Penanganan Outlier")
            ax2.set_ylabel("Nilai Normalisasi")
            ax2.set_xlabel("Index Data")
            st.pyplot(fig2)

        st.subheader('Setelah Penangan Outlier')
        col1, col2 = st.columns(2)
        with col1:
            data_bersih
        with col2:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=data_bersih, palette="tab10", dashes=False, ax=ax2)
            ax2.set_title("Setelah Penanganan Outlier")
            ax2.set_ylabel("Nilai Normalisasi")
            ax2.set_xlabel("Index Data")
            st.pyplot(fig2)


    with tab5:
        st.warning('NORMALISASI DATA | Dilakukan untuk menyeimbangkan nilai data sehingga jarak antar nilai tidak terlalu jauh dan seragam menggunakan min-max scaling')
        st.code("""
from sklearn.preprocessing import MinMaxScaler

data_prepro = data_imputed.values

# ======== Pisahkan fitur dan target ========
X = data_prepro[:, :-1]
y = data_prepro[:, -1].reshape(-1, 1)

# ======== Normalisasi MinMax ========
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
""")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Sebelum Normalisasi')
            fitur_df = pd.DataFrame(fitur, columns=variabel.columns[:-1])
            target_df = pd.DataFrame(target, columns=["Produksi"])
            data_before = pd.concat([fitur_df, target_df], axis=1).reset_index(drop=True)
            st.write("Nilai Minimum dan Maksimum Sebelum Normalisasi")
            st.dataframe(data_before.agg(['min', 'max']).T.rename(columns={'min': 'Min', 'max': 'Max'}))

            a, b = st.columns(2)
            with a:
                st.write('Fitur')
                fitur
            with b:
                st.write('Target')
                target

            st.write("Grafik Fitur & Target Sebelum Normalisasi")
            fig1, ax1 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=data_before, palette="tab10", dashes=False, ax=ax1)
            ax1.set_title("Sebelum Normalisasi (Skala Asli)")
            ax1.set_ylabel("Nilai Asli")
            ax1.set_xlabel("Index Data")
            st.pyplot(fig1)

        with col2:
            st.subheader('Setelah Normalisasi')
            fitur_norm_df = pd.DataFrame(X_scaled, columns=variabel.columns[:-1])
            target_norm_df = pd.DataFrame(y_scaled, columns=["Produksi"])
            data_after = pd.concat([fitur_norm_df, target_norm_df], axis=1).reset_index(drop=True)
            st.write("Nilai Minimum dan Maksimum Sebelum Normalisasi")
            st.dataframe(data_after.agg(['min', 'max']).T.rename(columns={'min': 'Min', 'max': 'Max'}))

            a, b = st.columns(2)
            with a:
                st.write('Fitur')
                X_scaled
            with b:
                st.write('Target')
                y_scaled

            st.write("Grafik Fitur & Target Setelah Normalisasi")
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            sns.lineplot(data=data_after, palette="tab10", dashes=False, ax=ax2)
            ax2.set_title("Setelah Normalisasi (Skala 0–1)")
            ax2.set_ylabel("Nilai Normalisasi")
            ax2.set_xlabel("Index Data")
            st.pyplot(fig2)


    with tab6:
        st.success('DATASET')
        st.write("Berikut merupakan data yang telah siap dilatih dan diuji menggunakan metode Support Vector Regression (SVR).")
        data_gabungan = pd.DataFrame(
            data = np.hstack((X_scaled, y_scaled.reshape(-1, 1))),
            columns = ['Kecamatan', 'Luas Tanam (Ha)', 'Luas Panen (Ha)', 'Curah Hujan (mm)', 'Produksi (Ton)']
        )
        data_gabungan



if (selected == 'Modelling'):
    tab_pelatihan, tab_pengujian = st.tabs(["Pelatihan Data", "Pengujian Data"])

    with tab_pelatihan:
        st.warning('Pelatihan Data')
        st.image("skenario uji coba.png", caption="Skenario Uji Coba", width=600)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Hasil pelatihan data berdasarkan skenario uji coba : ")
            results_df

        with col2:
            st.write("Diperoleh parameter terbaik :", best_params)

            # Prediksi ulang
            y_train_pred_scaled = best_model.predict(X_train).reshape(-1, 1)
            # Denormalisasi
            y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
            y_train_orig = scaler_y.inverse_transform(y_train)
            # Evaluasi
            mape_train = mean_absolute_percentage_error(y_train_orig, y_train_pred)
            st.info(f"MAPE : {mape_train * 100:.2f}%")

    with tab_pengujian:
        st.warning("Pengujian Data Berdasarkan Penggunaan Parameter Terbaik Dari Hasil Pelatihan Data")
        # Prediksi ulang
        y_test_pred_scaled = best_model.predict(X_test).reshape(-1, 1)
        # Denormalisasi
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)
        y_test_orig = scaler_y.inverse_transform(y_test)

        a, b = st.columns(2)
        with a:
            st.write("Perbandingan antara data aktual dan data peramalan produksi padi dari penggunaan model terbaik : ")
            hasil_peramalan = pd.DataFrame({
                'Data Aktual Produksi': y_test_orig.flatten(),
                'Data Peramalan Produksi': y_test_pred.flatten()
            })
            hasil_peramalan
        with b:
            st.write("Visualisasi perbandingan antara data aktual dan hasil prediksi produksi padi :")
            # Buat grafik
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(hasil_peramalan.index, hasil_peramalan['Data Aktual Produksi'], label='Data Aktual', color='orange')
            ax.plot(hasil_peramalan.index, hasil_peramalan['Data Peramalan Produksi'], label='Data Peramalan', color='red', marker='o', linewidth=2)
            ax.set_title('Perbandingan Data Aktual Produksi dan Data Peramalan Produksi')
            ax.set_xlabel('Index')
            ax.set_ylabel('Produksi')
            ax.legend()
            ax.grid(True)

            # Tampilkan di Streamlit
            st.pyplot(fig)

        # Evaluasi
        mape_test = mean_absolute_percentage_error(y_test_orig, y_test_pred)
        st.info(f"MAPE Data Pengujian : {mape_test * 100:.2f}%")

if (selected == 'Prediction'):
    simulasi, peramalan = st.tabs(["Simulasi", "Peramalan"])

    with simulasi:
        st.info("Prediksi produksi padi dengan memilih kecamatan serta input variabel luas tanam, luas panen dan curah hujan.")

        # Ambil nama kecamatan unik dari dataframe
        selected_kecamatan = st.selectbox("Pilih Kecamatan:",var_copy['Kecamatan'].unique())

        # Input numerik
        luas_tanam = st.number_input("Masukkan perkiraan luas tanam padi (Ha):", min_value=0, step=1)
        luas_panen = st.number_input("Masukkan perkiraan luas panen padi (Ha):", min_value=0, step=1)
        curah_hujan = st.number_input("Masukkan perkiraan curah hujan dalam setahun (mm):", min_value=0, step=1)

        if st.button("Prediksi"):
            kecamatan_encoded = label_encoder.transform([selected_kecamatan])[0]
            input_data = np.array([[kecamatan_encoded, luas_tanam, luas_panen, curah_hujan]])
            input_scaled = scaler_X.transform(input_data)
            hasil_scaled = best_model.predict(input_scaled).reshape(-1, 1)
            hasil_prediksi = scaler_y.inverse_transform(hasil_scaled)

            if hasil_prediksi > 0:
                st.success(f"Hasil prediksi produksi padi: {float(hasil_prediksi):.2f} ton")
            else:
                st.error(f"Hasil prediksi produksi padi: {float(hasil_prediksi):.2f} ton")

    with peramalan:
        st.info("Peramalan produksi padi tahun 2024 berdasarkan data historis rata-rata per kecamatan dari tahun 2018 hingga 2023.")

        # Gabungkan data imputasi dengan data asli agar dapat kolom Kabupaten
        data_gabung = data.copy()
        data_gabung['Luas Tanam (Ha)'] = data_imputed['Luas Tanam (Ha)']
        data_gabung['Luas Panen (Ha)'] = data_imputed['Luas Panen (Ha)']
        data_gabung['Curah Hujan (mm)'] = data_imputed['Curah Hujan (mm)']
        data_gabung['Produksi (Ton)'] = data_imputed['Produksi (Ton)']

        # Hitung rata-rata per kecamatan dan ambil 1 kabupaten untuk masing-masing kecamatan
        df_peramalan = data_gabung.groupby('Kecamatan', as_index=False).agg({
            'Kabupaten': 'first',
            'Luas Tanam (Ha)': 'mean',
            'Luas Panen (Ha)': 'mean',
            'Curah Hujan (mm)': 'mean',
            'Produksi (Ton)': 'mean'
        })

        # Tambahkan tahun peramalan
        df_peramalan['Tahun'] = 2024

        # Siapkan fitur peramalan
        fitur = ['Kecamatan', 'Luas Tanam (Ha)', 'Luas Panen (Ha)', 'Curah Hujan (mm)']
        target = 'Produksi (Ton)'

        # Transformasi label kecamatan ke angka (seperti saat training)
        df_peramalan['Kecamatan'] = label_encoder.transform(df_peramalan['Kecamatan'])

        # peramalan menggunakan model yang sudah dilatih
        X_pred = df_peramalan[fitur].values
        X_pred_scaled = scaler_X.transform(X_pred)
        y_pred_scaled = best_model.predict(X_pred_scaled).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        # Tambahkan hasil peramalan ke dataframe
        df_peramalan['Peramalan Produksi (ton)'] = y_pred.astype(int)

        # Kembalikan nama kecamatan asli
        df_peramalan['Kecamatan'] = label_encoder.inverse_transform(df_peramalan['Kecamatan'].astype(int))

        # Pilih kolom yang ditampilkan
        hasil_peramalan = df_peramalan[['Kabupaten', 'Kecamatan', 'Luas Tanam (Ha)', 'Luas Panen (Ha)', 'Curah Hujan (mm)', 'Peramalan Produksi (ton)']].copy()

        # Pilihan kecamatan
        kecamatan_options = ["Pilih Kecamatan"] + list(hasil_peramalan['Kecamatan'].unique())
        selected_kecamatan = st.selectbox("Pilih Kecamatan:", kecamatan_options)

        # Tampilkan hasil jika user memilih kecamatan
        if selected_kecamatan != "Pilih Kecamatan":
            hasil_kecamatan = hasil_peramalan[hasil_peramalan['Kecamatan'] == selected_kecamatan]

            st.write("### Hasil Peramalan Produksi Padi:")
            st.dataframe(hasil_kecamatan.reset_index(drop=True))
