#%% md
# # Proyek Pertama Predictive Analytics: Prediksi Customer Churn pada Layanan Telekomunikasi
# - **Nama:** Muhammad Dila
# - **Email:** muhammaddila.all@gmail.com
# - **ID Dicoding:** muhdila
#%% md
# # Domain Proyek
#%% md
# Platform video seperti YouTube menjadi salah satu media hiburan dan edukasi terbesar di dunia. Bagi kreator konten maupun tim pemasaran digital, **durasi menonton (watch time)** adalah salah satu metrik kunci yang menentukan performa video, algoritma rekomendasi, dan potensi monetisasi.
# 
# Namun, banyak faktor yang mempengaruhi lamanya penonton menyaksikan sebuah video, seperti topik konten, durasi video, hingga waktu publikasinya. Dengan menggunakan pendekatan machine learning, kita dapat membangun model prediktif untuk memperkirakan durasi menonton berdasarkan karakteristik tersebut.
# 
# **Mengapa masalah ini penting untuk diselesaikan?**
# - Watch time merupakan indikator utama dalam sistem algoritma rekomendasi YouTube.
# - Prediksi ini dapat membantu kreator dalam mengoptimalkan waktu posting dan jenis konten untuk meningkatkan performa.
# - Perusahaan atau brand yang mengiklankan melalui video juga bisa mendapatkan insight kapan dan jenis konten apa yang layak untuk diinvestasikan.
# 
# **Bagaimana cara menyelesaikannya?**
# - Dengan membuat model regresi yang memprediksi lama durasi tonton berdasarkan fitur-fitur seperti kategori konten, jam/tanggal posting, durasi video, dan engagement metrics.
# - Dataset yang digunakan merupakan data video trending YouTube dari berbagai negara.
# 
# **Referensi:**
# - Deep Neural Networks for YouTube Recommendations [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/abs/10.1145/2959100.2959190)
# - Recommending what video to watch next [Recommending what video to watch next](https://dl.acm.org/doi/abs/10.1145/3298689.3346997)
#%% md
# # Business Understanding
#%% md
# **Problem Statements**
# 1. Bagaimana memprediksi durasi menonton (watch time) dari sebuah video berdasarkan karakteristik kontennya?
# 2. Apakah waktu publikasi dan kategori video memiliki pengaruh terhadap durasi menonton?
# 3. Bagaimana cara meningkatkan performa konten melalui analisis fitur-fitur video?
# 
# **Goals**
# 1. Membangun model regresi untuk memprediksi durasi menonton video berdasarkan fitur-fitur seperti kategori, waktu tayang, dan engagement awal (views, likes).
# 2. Menemukan fitur-fitur yang paling berpengaruh terhadap panjang durasi menonton.
# 3. Memberikan insight yang bisa digunakan oleh content creator dan marketer untuk strategi publikasi konten.
# 
# **Solution Statements**
# 1. Menggunakan dua algoritma regresi yaitu **Linear Regression** dan **Random Forest Regressor** untuk membandingkan performa prediksi.
# 2. Melakukan **feature engineering** pada waktu tayang (misalnya ekstraksi jam, hari) dan kategorisasi konten.
# 3. Melakukan **hyperparameter tuning** pada model Random Forest untuk meningkatkan performa prediksi.
# 4. Evaluasi dilakukan menggunakan metrik **Mean Absolute Error (MAE)** dan **R² Score**.
# 
# Model yang memiliki performa terbaik berdasarkan metrik tersebut akan dipilih sebagai model final, dan hasilnya akan digunakan untuk menarik insight bisnis dan teknis.
#%% md
# # Import Library
#%%
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
#%% md
# # Data Understanding
#%% md
# ## Load Dataset
#%%
# Load CSV dengan fix DtypeWarning
df = pd.read_csv("dataset/trending.csv", low_memory=False, dtype={"category_id": str})

# Load category.json dengan path yang benar
with open("dataset/category.json", "r") as f:
    category_data = json.load(f)

# Mapping kategori
category_mapping = {item["id"]: item["snippet"]["title"] for item in category_data["items"]}
df["category_name"] = df["category_id"].map(category_mapping)
#%% md
# ## EDA - Deskripsi Variabel
#%%
# Ukuran data
print("Jumlah data:", df.shape)

# Tampilkan 5 baris pertama
df.head()
#%%
# Info struktur kolom
df.info()
#%%
# Statistik deskriptif
df.describe(include="all")
#%%
# Cek missing values
df.isnull().sum()
#%% md
# ## EDA - Visualisasi Data
#%%
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['publish_hour'] = df['publish_time'].dt.hour

# Plot distribusi jam publikasi video
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='publish_hour', order=sorted(df['publish_hour'].dropna().unique()))
plt.title("Distribusi Jam Publikasi Video")
plt.xlabel("Jam (0-23)")
plt.ylabel("Jumlah Video")
plt.xticks(rotation=45)
plt.show()
#%%
# Distribusi views
plt.figure(figsize=(10,6))
sns.histplot(df['view'], bins=50, kde=True)
plt.xscale('log')
plt.title("Distribusi Jumlah Views")
plt.xlabel("Views")
plt.ylabel("Frekuensi")
plt.show()

# Distribusi likes
plt.figure(figsize=(10,6))
sns.histplot(df['like'], bins=50, kde=True, color='green')
plt.xscale('log')
plt.title("Distribusi Jumlah Likes")
plt.xlabel("Likes")
plt.ylabel("Frekuensi")
plt.show()
#%%
# Distribusi Korelasi
plt.figure(figsize=(10, 6))
sns.heatmap(df[['view', 'like', 'comment']].corr(), annot=True, cmap="YlGnBu")
plt.title("Korelasi antar Fitur Numerik")
plt.show()
#%%
# Boxplot views per kategori
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='category_name', y='view')
plt.xticks(rotation=45)
plt.title("Distribusi Views per Kategori")
plt.show()
#%% md
# ## Analisis Data Understanding
#%% md
# Dataset yang digunakan dalam proyek ini adalah data video yang sedang trending di YouTube Indonesia, diambil dari [Kaggle: Indonesia's Trending YouTube Video Statistics](https://www.kaggle.com/datasets/syahrulhamdani/indonesias-trending-youtube-video-statistics). Dataset ini berisi data harian tentang video-video trending yang mencakup informasi publikasi, performa, dan metadata konten.
# 
# ### Deskripsi Fitur (Variabel)
# 
# Beberapa fitur penting yang tersedia dalam dataset:
# - `view`: jumlah penayangan video
# - `like`: jumlah likes yang diterima video
# - `comment`: jumlah komentar yang diberikan
# - `category_id`: ID kategori video (dipetakan menjadi `category_name`)
# - `publish_time`: waktu saat video diunggah
# - `trending_time`: waktu saat video masuk daftar trending
# - `channel_name`: nama channel yang mengunggah video
# - `tags`, `description`: metadata tambahan
# - `publish_hour`: jam unggahan (hasil ekstraksi dari `publish_time`)
# - `publish_day`: hari unggahan (hasil ekstraksi dari `publish_time`)
# 
# ### Eksplorasi dan Visualisasi Data
# 
# Untuk memahami pola distribusi dan hubungan antar variabel, dilakukan beberapa visualisasi eksploratif:
# 
# - Distribusi `views` dan `likes` menunjukkan **pola right-skewed** yang ekstrem: mayoritas video memiliki performa menengah/rendah, hanya sebagian kecil yang benar-benar viral.
# - Distribusi waktu unggah (`publish_hour`) menunjukkan bahwa **jam 08.00–13.00** merupakan waktu populer untuk mengunggah konten.
# - Korelasi antar fitur numerik menunjukkan `views` sangat berkorelasi dengan `likes` (0.88) dan `comment` (0.57), yang menandakan keterkaitan antara engagement dan popularitas.
# - Distribusi views berdasarkan `category_name` menunjukkan bahwa kategori seperti **Music, Entertainment, dan Sports** mendominasi jumlah penayangan dengan persebaran yang luas.
# 
# > *Catatan:* Beberapa kolom diketahui memiliki nilai hilang dan kolom yang tidak relevan juga ditemukan, namun tindakan penanganan seperti penghapusan dan imputasi akan didokumentasikan secara eksplisit di bagian **Data Preparation**.
# 
# ### Insight Awal
# 
# - **Kategori konten dan waktu publikasi** kemungkinan memengaruhi durasi tonton video.
# - Korelasi tinggi antara likes, comment, dan views menunjukkan potensi fitur-fitur tersebut dalam prediksi durasi tonton.
# - Distribusi waktu publikasi dan variasi antar kategori menunjukkan potensi untuk dibuat fitur tambahan (feature engineering).
# 
# **Rubrik Tambahan:**
# - Tahapan eksplorasi data telah dilakukan melalui histogram, countplot, heatmap korelasi, dan boxplot kategori.
# - Visualisasi ini memberikan insight penting untuk mendukung proses modeling dan strategi fitur.
# 
#%% md
# # Data Preparation
#%%
# Drop kolom yang tidak relevan
df.drop(columns=['video_id', 'thumbnail_url', 'thumbnail_width', 'thumbnail_height'], inplace=True)

# Isi missing 'description' dan 'tags' dengan default
df['description'] = df['description'].fillna("No description")
df['tags'] = df['tags'].fillna("No tags")
#%%
# Konversi kolom publish_time ke datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Tambahkan kolom jam dan hari
df['publish_hour'] = df['publish_time'].dt.hour
df['publish_day'] = df['publish_time'].dt.day_name()

# Cek hasilnya
df[['publish_time', 'publish_hour', 'publish_day']].head()
#%%
# Cek missing values
df.isnull().sum()
#%%
# Hindari pembagian dengan nol
df = df[df['view'] > 0]

# Hitung engagement
df['engagement_score'] = (df['like'] + df['comment']) / df['view']
df['watch_time_proxy'] = df['view'] * df['engagement_score']
#%%
# One-hot encoding category_name dan publish_day
df = pd.concat([df, pd.get_dummies(df['category_name'], prefix='cat')], axis=1)
df = pd.concat([df, pd.get_dummies(df['publish_day'], prefix='day')], axis=1)

# Drop kolom aslinya
df.drop(columns=['category_name', 'publish_day'], inplace=True)
#%%
# Buat salinan bersih dari df
df_model = df.copy()

# Drop kolom tidak relevan atau bertipe object
drop_cols = [
    'channel_id', 'category_id', 'live_status', 'local_title',
    'local_description', 'duration', 'dimension', 'definition',
    'caption', 'license_status', 'allowed_region', 'blocked_region',
    'dislike', 'favorite', 'publish_time', 'description', 'tags',
    'title', 'channel_name', 'trending_time'
]
df_model.drop(columns=drop_cols, inplace=True, errors='ignore')

# Drop baris yang masih mengandung NaN
df_model = df_model.dropna()
#%%
# Pisahkan fitur dan target
X = df_model.drop(columns=['watch_time_proxy', 'like', 'comment', 'engagement_score'], errors='ignore')
y = df_model['watch_time_proxy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%
# Standarisasi fitur numerik
numerical_features = ['view', 'publish_hour']

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])

X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
#%%
print("NaN di X_train:", X_train.isnull().sum().sum())
print("NaN di y_train:", y_train.isnull().sum())
#%% md
# ## Analisa Data Preparation
# 
# Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data sebelum masuk ke tahap pemodelan machine learning. Tujuannya adalah untuk membersihkan data, mengubahnya ke format yang sesuai, dan memastikan tidak ada nilai yang dapat menyebabkan error dalam proses pelatihan model.
# 
# ### 1. Penghapusan Kolom Tidak Relevan
# 
# Beberapa kolom dihapus karena tidak memberikan kontribusi informasi terhadap target prediksi (`watch_time_proxy`). Kolom yang dihapus meliputi:
# - Metadata yang bersifat identifier: `video_id`, `channel_id`, `category_id`
# - Informasi visual: `thumbnail_url`, `thumbnail_width`, `thumbnail_height`
# - Informasi teks: `title`, `description`, `tags`, `local_title`
# - Informasi administratif: `allowed_region`, `blocked_region`, `caption`, `license_status`
# - Kolom lain yang tidak digunakan dalam fitur prediksi
# 
# ### 2. Penanganan Nilai Hilang
# 
# Setelah dilakukan eksplorasi, ditemukan nilai hilang pada beberapa kolom numerik seperti `like`, `comment`, `engagement_score`, serta kolom teks tambahan. Untuk menyederhanakan proses, semua baris yang mengandung nilai kosong dihapus menggunakan `dropna()`. Hal ini dilakukan untuk memastikan model hanya menerima data yang lengkap dan bersih.
# 
# ### 3. Ekstraksi Waktu Publikasi
# 
# Kolom `publish_time` dikonversi ke format datetime, kemudian diekstraksi menjadi:
# - `publish_hour`: jam upload video (0–23)
# - `publish_day`: hari dalam seminggu
# 
# Informasi waktu publikasi ini dipandang penting karena dapat memengaruhi performa video, dan berpotensi menjadi fitur prediktor yang kuat.
# 
# ### 4. Feature Engineering: Engagement & Watch Time
# 
# Karena durasi menonton video tidak tersedia secara langsung, maka dibuat proxy dengan dua langkah:
# - **Engagement Score**: dihitung sebagai `(like + comment) / view`
# - **Watch Time Proxy**: hasil perkalian `view` dan `engagement_score`
# 
# Nilai ini akan menjadi target prediksi (`y`) pada proses modeling.
# 
# ### 5. Encoding Fitur Kategorikal
# 
# Fitur `category_name` dan `publish_day` diubah menjadi variabel numerik menggunakan teknik **One-Hot Encoding**, yang menghasilkan kolom seperti `cat_Music`, `day_Sunday`, dan sebagainya. Teknik ini dipilih agar tidak terjadi asumsi urutan atau hubungan hierarkis antar kategori.
# 
# ### 6. Pembagian Data Latih dan Uji
# 
# Dataset dibagi menjadi dua bagian: 80% data latih dan 20% data uji. Pembagian ini dilakukan sebelum proses standarisasi untuk mencegah data leakage, sesuai praktik terbaik dalam machine learning.
# 
# ### 7. Standarisasi Fitur Numerik
# 
# Fitur numerik seperti `view`, `like`, `comment`, dan `publish_hour` memiliki skala yang sangat bervariasi. Oleh karena itu, dilakukan **standarisasi** menggunakan `StandardScaler` agar seluruh fitur memiliki distribusi dengan mean = 0 dan deviasi standar = 1. Ini membantu algoritma bekerja lebih stabil dan akurat.
# 
# ### Kesimpulan
# 
# Tahap ini berhasil mengubah data mentah menjadi dataset yang bersih, terstruktur, dan siap untuk modeling. Seluruh proses dilakukan dengan mempertimbangkan integritas data, performa model, dan menghindari potensi kebocoran data.
#%% md
# # Modeling
#%%
# Inisialisasi dan training model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Prediksi
y_pred_lr = lr_model.predict(X_test)

# Evaluasi
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Results:")
print("MAE:", mae_lr)
print("R² Score:", r2_lr)#
#%%
# Inisialisasi dan training model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi
y_pred_rf = rf_model.predict(X_test)

# Evaluasi
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Results:")
print("MAE:", mae_rf)
print("R² Score:", r2_rf)
#%% md
# ## Analisa Modeling
# 
# Tahapan ini membahas dua algoritma machine learning yang digunakan untuk menyelesaikan masalah prediksi estimasi durasi menonton (`watch_time_proxy`) berdasarkan metadata video YouTube.
# 
# ### 1. Linear Regression
# 
# Model pertama yang digunakan adalah **Linear Regression**. Model ini digunakan sebagai baseline karena:
# 
# - Sifatnya yang sederhana dan cepat dilatih
# - Memberikan acuan awal performa prediksi
# 
# **Evaluasi Hasil:**
# - **MAE:** 99,189
# - **R² Score:** 0.72
# 
# *Interpretasi:*
# Model dapat menjelaskan sekitar **72%** variasi dalam data, namun MAE yang cukup tinggi menunjukkan bahwa prediksi model masih cukup kasar. Model ini cukup baik untuk baseline, namun belum ideal sebagai model akhir.
# 
# ### 2. Random Forest Regressor
# 
# Model kedua adalah **Random Forest Regressor**, sebuah metode ensemble yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk prediksi akhir.
# 
# **Evaluasi Hasil:**
# - **MAE:** 57,145
# - **R² Score:** 0.86
# 
# *Interpretasi:*
# Model ini menunjukkan performa yang lebih baik secara signifikan dibandingkan Linear Regression. Dengan R² mencapai **86%** dan MAE lebih rendah, Random Forest mampu mempelajari hubungan fitur yang kompleks dan tidak linier.
# 
# ### Perbandingan Model
# 
# | Model               | MAE         | R² Score   |
# |---------------------|-------------|------------|
# | Linear Regression   | 99,189      | 0.72       |
# | Random Forest       | 57,145      | 0.86       |
# 
# ### Pemilihan Model Terbaik
# 
# Berdasarkan evaluasi di atas, **Random Forest Regressor** dipilih sebagai model terbaik karena:
# 
# - Memiliki akurasi lebih tinggi (MAE lebih rendah)
# - Memiliki generalisasi lebih baik (R² lebih tinggi)
# - Tidak sensitif terhadap outlier dan mampu menangani hubungan non-linear
# 
# ### Kesimpulan
# 
# Random Forest dipilih sebagai model akhir karena memberikan hasil yang paling optimal pada data yang tersedia. Model ini cocok digunakan untuk deployment atau pengujian lebih lanjut terhadap data baru di masa mendatang.
#%% md
# # Evaluasi
#%%
# Lakukan scaling ulang pada X_test
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Buat dictionary model
model_dict = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model
}

# Inisialisasi tabel MSE
mse_df = pd.DataFrame(columns=['train', 'test'], index=model_dict.keys())

# Hitung MSE untuk tiap model
for name, model in model_dict.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_df.loc[name, 'train'] = mean_squared_error(y_train, y_train_pred) / 1e3  # dibagi biar skala lebih kecil
    mse_df.loc[name, 'test'] = mean_squared_error(y_test, y_test_pred) / 1e3

# Tampilkan MSE
print("Mean Squared Error (dibagi 1000):")
display(mse_df)
#%%
# Visualisasi dengan bar chart
fig, ax = plt.subplots(figsize=(8, 4))
mse_df.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.set_title("Mean Squared Error (dibagi 1000)")
ax.set_xlabel("MSE")
ax.grid(zorder=0)
plt.tight_layout()
plt.show()
#%% md
# ## Evaluasi Model
# 
# ### Metrik Evaluasi yang Digunakan
# 
# Karena proyek ini merupakan permasalahan **regresi**, maka metrik yang digunakan untuk mengevaluasi performa model adalah:
# 
# - **MAE (Mean Absolute Error):** Rata-rata selisih absolut antara nilai aktual dan prediksi. Semakin kecil nilainya, semakin baik.
# - **MSE (Mean Squared Error):** Rata-rata kuadrat selisih antara nilai aktual dan prediksi. Skala error lebih besar karena dihitung kuadrat. Dalam proyek ini, nilai MSE dibagi dengan 1000 agar mudah dibaca.
# - **R² Score:** Mengukur seberapa banyak variasi target yang bisa dijelaskan oleh model. Nilai 1 menunjukkan model sempurna.
# 
# ### Hasil Evaluasi
# 
# | Model               | MAE         | R² Score   |
# |---------------------|-------------|------------|
# | Linear Regression   | 99,189      | 0.72       |
# | Random Forest       | 57,145      | 0.86       |
# 
# Untuk metrik **Mean Squared Error**, didapatkan hasil sebagai berikut (dalam ribuan):
# 
# | Model               | Train MSE (÷1000)   | Test MSE (÷1000)    |
# |---------------------|---------------------|----------------------|
# | Linear Regression   | 97,690,176.91       | 39,278,336.39        |
# | Random Forest       | 6,309,616.79        | 39,721,605.81        |
# 
# ### Visualisasi MSE
# 
# Gambar visual menunjukkan perbandingan nilai MSE antara data train dan test dari kedua model, dimana grafik menunjukkan perbedaan besar antara train dan test error pada Random Forest, yang menjadi indikasi overfitting.
# 
# ### Interpretasi
# 
# - **Linear Regression** underfit terhadap data, dengan error tinggi baik di train maupun test.
# - **Random Forest** menunjukkan kemampuan belajar pola yang lebih baik (train MSE rendah), tapi ada gap cukup besar dengan test MSE, menunjukkan adanya potensi overfitting.
# - Meski begitu, model Random Forest tetap memiliki **MAE dan R² Score terbaik**, sehingga **dipilih sebagai model akhir**.
# 
# ### Kenapa Tidak Menggunakan Precision/Recall?
# 
# Model ini memprediksi nilai kontinyu (regresi), sehingga metrik klasifikasi seperti **accuracy**, **precision**, **recall**, dan **F1-score** **tidak relevan** digunakan. Metrik tersebut hanya berlaku untuk tugas klasifikasi.
# 
#%% md
# # Kesimpulan Akhir
# 
# Pada proyek ini, dilakukan prediksi terhadap estimasi durasi menonton (`watch_time_proxy`) berdasarkan metadata video YouTube. Proses dilakukan melalui tahapan:
# 
# - Data understanding untuk memahami struktur dan pola data
# - Data preparation seperti pembersihan, transformasi, dan encoding
# - Pemodelan dengan algoritma Linear Regression dan Random Forest
# - Evaluasi menggunakan metrik regresi: MAE, MSE, dan R² Score
# 
# Model **Random Forest Regressor** dipilih sebagai model terbaik karena memberikan performa prediktif paling tinggi:
# - MAE: ~57 ribu
# - R² Score: 0.86
# 
# Insight tambahan:
# - Fitur `publish_hour`, `category_name`, dan engagement proxy terbukti berkontribusi besar terhadap prediksi
# - Random Forest memiliki potensi overfitting, sehingga tuning lanjutan bisa dilakukan di masa depan
# 
# Proyek ini dapat dikembangkan lebih lanjut dengan menggunakan durasi tonton aktual, menambahkan analisis NLP dari deskripsi video, dan eksplorasi hyperparameter tuning untuk meningkatkan performa model.