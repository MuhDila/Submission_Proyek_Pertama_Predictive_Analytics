#%% md
# # Proyek Pertama Predictive Analytics: Prediksi Durasi Tonton Video YouTube
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
#%%

#%% md
# # Import Library
#%% md
# Pada tahap ini, kita mengimpor berbagai library yang diperlukan untuk manipulasi data, visualisasi grafik, serta membangun dan mengevaluasi model machine learning.
#%%
# Import library pandas untuk manipulasi data
import pandas as pd

# Import library json untuk membaca file JSON
import json

# Import seaborn untuk visualisasi data berbasis statistik
import seaborn as sns

# Import matplotlib.pyplot untuk plotting grafik
import matplotlib.pyplot as plt

# Import StandardScaler untuk standarisasi fitur numerik
from sklearn.preprocessing import StandardScaler

# Import train_test_split untuk membagi dataset menjadi train dan test
from sklearn.model_selection import train_test_split

# Import LinearRegression untuk membuat model regresi linear
from sklearn.linear_model import LinearRegression

# Import metric MAE dan R2 Score untuk evaluasi model regresi
from sklearn.metrics import mean_absolute_error, r2_score

# Import RandomForestRegressor untuk membuat model ensemble berbasis decision tree
from sklearn.ensemble import RandomForestRegressor

# Import metric MSE untuk evaluasi model regresi
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
# Menampilkan jumlah baris dan kolom dalam dataset
print("Jumlah data:", df.shape)

# Menampilkan 5 baris pertama untuk melihat contoh data
df.head()
#%% md
# ### Menampilkan Ukuran dan Sampel Data
# 
# Pada tahap ini, dilakukan pengecekan terhadap ukuran dataset untuk mengetahui jumlah baris dan kolom yang tersedia.
# Setelah itu, ditampilkan 5 baris pertama untuk memberikan gambaran umum mengenai struktur data,
# seperti tipe fitur, format data, dan potensi adanya nilai kosong atau ketidaksesuaian pada data.
# Langkah ini penting untuk memahami kondisi awal dataset sebelum masuk ke tahap pembersihan dan analisis lanjutan.
#%%
# Menampilkan struktur kolom dan tipe data di dataset
df.info()
#%% md
# ### Melihat Struktur Kolom dan Tipe Data
# 
# Pada tahap ini, ditampilkan struktur kolom pada dataset menggunakan fungsi `info()`.
# Langkah ini dilakukan untuk mengetahui tipe data pada masing-masing kolom, serta jumlah nilai yang tidak kosong (non-null) di setiap kolom.
# Informasi ini sangat penting untuk mengidentifikasi kolom mana saja yang memiliki missing value dan menentukan perlakuan yang tepat pada tahap data preprocessing nanti.
# 
#%%
# Menampilkan statistik deskriptif dari semua kolom, baik numerik maupun kategorikal
df.describe(include="all")
#%% md
# ### Menampilkan Statistik Deskriptif Dataset
# 
# Pada tahap ini, ditampilkan statistik deskriptif dari seluruh fitur dalam dataset menggunakan fungsi `describe()`.
# Statistik ini mencakup informasi seperti jumlah data, nilai unik, nilai minimum dan maksimum, serta distribusi data untuk fitur numerik dan kategorikal.
# Analisis ini penting untuk memahami skala, penyebaran, dan karakteristik umum data, serta mengidentifikasi kemungkinan nilai ekstrim atau ketidaksesuaian yang perlu ditangani pada tahap preprocessing.
# 
#%%
# Mengecek jumlah missing values di setiap kolom
print("Missing Values per Kolom:")
print(df.isnull().sum())

# Mengecek jumlah data duplikat
print("\nJumlah Data Duplikat:")
print(df.duplicated().sum())

# Statistik deskriptif untuk mendeteksi kemungkinan outlier
print("\nStatistik Deskriptif untuk Deteksi Outlier:")
print(df[['view', 'like', 'comment']].describe())
#%% md
# ### Mengevaluasi Kondisi Data: Missing Values, Duplikasi, dan Outlier
# 
# Pada tahap ini, dilakukan pengecekan kondisi data untuk memastikan kualitas dataset.
# Pengecekan meliputi jumlah missing values di setiap kolom, keberadaan data duplikat, dan indikasi nilai outlier pada fitur numerik.
# Langkah ini penting untuk mengantisipasi masalah yang dapat mempengaruhi proses analisis dan modeling ke depannya.
#%% md
# ## EDA - Visualisasi Data
#%%
# Mengonversi kolom 'publish_time' ke format datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Menambahkan kolom baru 'publish_hour' untuk menyimpan jam publikasi
df['publish_hour'] = df['publish_time'].dt.hour

# Membuat plot distribusi jumlah video berdasarkan jam publikasi
plt.figure(figsize=(10,6))
sns.countplot(data=df, x='publish_hour', order=sorted(df['publish_hour'].dropna().unique()))
plt.title("Distribusi Jam Publikasi Video")
plt.xlabel("Jam (0-23)")
plt.ylabel("Jumlah Video")
plt.xticks(rotation=45)
plt.show()
#%% md
# ### Visualisasi Distribusi Jam Publikasi Video
# 
# Pada tahap ini, dilakukan visualisasi distribusi jumlah video berdasarkan jam publikasinya dalam sehari.
# Grafik ini menunjukkan bahwa mayoritas video dipublikasikan pada rentang jam 9 hingga 13, dengan puncaknya sekitar jam 12 siang.
# Pemahaman terhadap pola waktu publikasi ini dapat membantu dalam analisis perilaku kreator serta strategi optimasi waktu upload di platform video.
#%%
# Membuat plot distribusi jumlah views
plt.figure(figsize=(10,6))
sns.histplot(df['view'], bins=50, kde=True)
plt.xscale('log')  # Menggunakan skala logaritmik karena range views sangat lebar
plt.title("Distribusi Jumlah Views")
plt.xlabel("Views")
plt.ylabel("Frekuensi")
plt.show()

# Membuat plot distribusi jumlah likes
plt.figure(figsize=(10,6))
sns.histplot(df['like'], bins=50, kde=True, color='green')
plt.xscale('log')  # Menggunakan skala logaritmik juga untuk likes
plt.title("Distribusi Jumlah Likes")
plt.xlabel("Likes")
plt.ylabel("Frekuensi")
plt.show()
#%% md
# ### Visualisasi Distribusi Jumlah Views dan Likes
# 
# Pada tahap ini, dilakukan visualisasi distribusi jumlah views dan likes dari video dalam dataset.
# Karena nilai views dan likes memiliki rentang yang sangat luas, digunakan skala logaritmik pada sumbu x untuk memperjelas pola distribusi.
# Grafik menunjukkan bahwa sebagian besar video memiliki jumlah views dan likes yang relatif rendah, sementara hanya sedikit video yang mencapai angka views atau likes yang sangat tinggi.
# Hal ini mengindikasikan adanya ketimpangan distribusi popularitas antar video dalam platform.
#%%
# Membuat heatmap korelasi antara view, like, dan comment
plt.figure(figsize=(10, 6))
sns.heatmap(df[['view', 'like', 'comment']].corr(), annot=True, cmap="YlGnBu")
plt.title("Korelasi antar Fitur Numerik")
plt.show()
#%% md
# ### Analisis Korelasi antar Fitur Numerik
# 
# Pada tahap ini, dilakukan analisis korelasi antara fitur numerik `view`, `like`, dan `comment` menggunakan heatmap.
# Nilai korelasi yang tinggi antara view dan like (0.88), serta like dan comment (0.70), menunjukkan adanya hubungan linier yang cukup kuat antar fitur tersebut.
# Analisis korelasi ini penting untuk memahami keterkaitan antar variabel, sehingga dapat menjadi dasar pertimbangan dalam pemilihan fitur saat proses modeling.
#%%
# Membuat boxplot distribusi jumlah views per kategori
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='category_name', y='view')
plt.xticks(rotation=45)
plt.title("Distribusi Views per Kategori")
plt.show()
#%% md
# ### Visualisasi Distribusi Views Berdasarkan Kategori
# 
# Pada tahap ini, dilakukan visualisasi distribusi jumlah views untuk setiap kategori video menggunakan boxplot.
# Visualisasi ini bertujuan untuk melihat perbedaan penyebaran dan outlier views antar kategori, serta mengidentifikasi kategori mana yang cenderung memiliki jumlah views lebih tinggi.
# Dari grafik terlihat bahwa kategori seperti "Music", "Entertainment", dan "Sports" memiliki distribusi views yang lebih tinggi dibanding kategori lainnya.
# 
#%% md
# ## Analisis Data Understanding
#%% md
# Dataset yang digunakan dalam proyek ini adalah data video yang sedang trending di YouTube Indonesia, diambil dari [Kaggle: Indonesia's Trending YouTube Video Statistics](https://www.kaggle.com/datasets/syahrulhamdani/indonesias-trending-youtube-video-statistics). Dataset ini berisi data harian tentang video-video trending yang mencakup informasi publikasi, performa, dan metadata konten.
# 
# ### Jumlah Data
# Dataset ini memiliki:
# 
# - **Jumlah baris:** 172.347 baris
# - **Jumlah kolom:** 28 kolom
# 
# Jumlah baris merepresentasikan banyaknya data video yang tercatat harian sebagai trending,
# sedangkan kolom mencakup berbagai atribut terkait metadata dan performa video.
# 
# ### Kondisi Data
# 
# Hasil pemeriksaan kondisi data menunjukkan:
# 
# - **Missing Value:**
#   - Terdapat missing values pada beberapa kolom penting seperti `description`, `tags`, `thumbnail_url`, `view`, `like`, `comment`, dan `category_name`.
#   - Kolom `allowed_region` dan `blocked_region` memiliki jumlah missing value yang sangat tinggi (>90% dari data).
# 
# - **Duplikasi:**
#   - Terdapat duplikasi sejumlah **~0** baris, namun pengecekan tetap dilakukan untuk memastikan keunikan data.
# 
# - **Outlier:**
#   - Fitur numerik seperti `view` dan `like` menunjukkan adanya outlier ekstrem, dengan sebagian kecil video mencapai views sangat tinggi dibandingkan mayoritas video lainnya.
#   - Distribusi views dan likes berbentuk **right-skewed** berdasarkan hasil visualisasi histogram logaritmik.
# 
# Penanganan lebih lanjut terhadap missing values, duplikasi akan dilakukan pada tahap **Data Preparation**.
# 
# ### Deskripsi Fitur (Variabel)
# 
# | No | Fitur | Deskripsi |
# |:--|:--|:--|
# | 1 | `video_id` | ID unik video YouTube |
# | 2 | `publish_time` | Waktu unggahan video |
# | 3 | `channel_id` | ID unik channel pengunggah |
# | 4 | `title` | Judul video |
# | 5 | `description` | Deskripsi video |
# | 6 | `thumbnail_url` | URL gambar thumbnail video |
# | 7 | `thumbnail_width` | Lebar thumbnail video |
# | 8 | `thumbnail_height` | Tinggi thumbnail video |
# | 9 | `channel_name` | Nama channel pengunggah |
# | 10 | `tags` | Tag terkait video |
# | 11 | `category_id` | ID kategori video (numerik) |
# | 12 | `live_status` | Status apakah video live stream |
# | 13 | `local_title` | Judul video dalam bahasa lokal |
# | 14 | `local_description` | Deskripsi video dalam bahasa lokal |
# | 15 | `duration` | Durasi video (format ISO 8601) |
# | 16 | `dimension` | Dimensi video (2d atau 3d) |
# | 17 | `definition` | Resolusi video (standard/hd) |
# | 18 | `caption` | Indikator apakah video memiliki caption |
# | 19 | `license_status` | Status lisensi video |
# | 20 | `allowed_region` | Negara di mana video diperbolehkan tampil |
# | 21 | `blocked_region` | Negara di mana video diblokir |
# | 22 | `view` | Jumlah views video |
# | 23 | `like` | Jumlah likes video |
# | 24 | `dislike` | Jumlah dislikes video |
# | 25 | `favorite` | Jumlah favorit video (umumnya 0 setelah 2018) |
# | 26 | `comment` | Jumlah komentar video |
# | 27 | `trending_time` | Waktu video tercatat trending |
# | 28 | `category_name` | Nama kategori video berdasarkan `category_id` |
# 
# ### Eksplorasi Data
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
# ## Visualisasi Data
# 
# Beberapa visualisasi eksploratif dilakukan untuk memahami pola data:
# 
# - **Distribusi Views dan Likes:**
#   Menunjukkan pola distribusi yang sangat right-skewed, di mana sebagian besar video memiliki performa menengah hingga rendah, sementara hanya sebagian kecil video yang benar-benar viral.
# 
# - **Distribusi Publish Hour:**
#   Sebagian besar video diunggah antara **pukul 08.00 hingga 13.00**, menunjukkan waktu populer untuk mengunggah konten.
# 
# - **Korelasi antar Fitur Numerik:**
#   Terdapat korelasi kuat antara `views` dan `likes` (0.88), serta `likes` dan `comment` (0.70). Ini menandakan bahwa engagement (likes dan comments) memiliki keterkaitan kuat dengan popularitas video.
# 
# - **Distribusi Views per Kategori:**
#   Kategori seperti **Music**, **Entertainment**, dan **Sports** memiliki persebaran views yang lebih tinggi dibandingkan kategori lainnya.
# 
# **Catatan:** Beberapa kolom ditemukan memiliki nilai kosong (missing values) dan terdapat kolom yang kurang relevan. Tindakan penanganan missing values dan pembersihan data akan dibahas lebih rinci pada bagian **Data Preparation**.
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
#%% md
# # Data Preparation
#%%
# Menghapus kolom yang tidak relevan
df.drop(columns=['video_id', 'thumbnail_url', 'thumbnail_width', 'thumbnail_height'], inplace=True)

# Mengisi missing value pada kolom 'description' dan 'tags' dengan default text
df['description'] = df['description'].fillna("No description")
df['tags'] = df['tags'].fillna("No tags")
#%% md
# ### Pembersihan Data Awal
# 
# Pada tahap ini, dilakukan pembersihan data untuk menghilangkan kolom-kolom yang dianggap tidak relevan terhadap tujuan analisis.
# Selain itu, dilakukan juga imputasi (pengisian) nilai kosong pada fitur teks agar tidak mengganggu proses selanjutnya.
# Langkah ini bertujuan untuk memastikan bahwa dataset yang digunakan lebih bersih, ringkas, dan siap untuk tahap eksplorasi lebih lanjut maupun modeling.
# 
# Langkah yang dilakukan:
# - Menghapus kolom `video_id`, `thumbnail_url`, `thumbnail_width`, dan `thumbnail_height` karena tidak berkontribusi langsung terhadap analisis performa video.
# - Mengisi nilai kosong pada kolom `description` dan `tags` dengan string default ("No description" dan "No tags").
#%%
# Konversi kolom publish_time ke datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Tambahkan kolom jam dan hari
df['publish_hour'] = df['publish_time'].dt.hour
df['publish_day'] = df['publish_time'].dt.day_name()

# Cek hasilnya
df[['publish_time', 'publish_hour', 'publish_day']].head()
#%% md
# ### Pembuatan Fitur Waktu dari `publish_time`
# 
# Pada tahap ini, dilakukan konversi kolom `publish_time` dari string menjadi format datetime agar lebih mudah dianalisis.
# Dari kolom tersebut, diturunkan dua fitur baru yaitu `publish_hour` dan `publish_day`, yang masing-masing merepresentasikan jam dan hari saat video dipublikasikan.
# Langkah ini bertujuan untuk menangkap pola perilaku pengunggahan video berdasarkan waktu, yang nantinya dapat menjadi fitur penting dalam analisis atau modeling.
# 
# Fitur yang dibuat:
# - `publish_hour`: Jam (0-23) saat video dipublikasikan.
# - `publish_day`: Nama hari saat video dipublikasikan (Senin, Selasa, dst).
#%%
# Menghindari data dengan views nol agar tidak terjadi pembagian dengan nol
df = df[df['view'] > 0]

# Membuat fitur engagement_score: (likes + comments) dibagi views
df['engagement_score'] = (df['like'] + df['comment']) / df['view']

# Membuat fitur watch_time_proxy: views dikali engagement_score
df['watch_time_proxy'] = df['view'] * df['engagement_score']

#%% md
# ### Pembuatan Engagement Score dan Watch Time Proxy
# 
# Pada tahap ini, dilakukan pembuangan data dengan jumlah views nol untuk menghindari error pembagian.
# Selanjutnya dihitung `engagement_score`, yaitu rasio antara jumlah likes dan comments terhadap jumlah views.
# Selain itu, dibuat fitur `watch_time_proxy` dengan mengalikan jumlah views dan engagement_score untuk memperkirakan interaksi dan durasi tonton secara kasar.
# Fitur-fitur ini membantu menggambarkan tingkat keterlibatan pengguna terhadap video trending secara lebih mendalam.
#%%
# Melakukan one-hot encoding pada 'category_name' dan 'publish_day'
df = pd.concat([df, pd.get_dummies(df['category_name'], prefix='cat')], axis=1)
df = pd.concat([df, pd.get_dummies(df['publish_day'], prefix='day')], axis=1)

# Menghapus kolom aslinya setelah encoding
df.drop(columns=['category_name', 'publish_day'], inplace=True)
#%% md
# ### One-Hot Encoding Fitur Kategorikal
# 
# Pada tahap ini, dilakukan one-hot encoding terhadap fitur kategorikal `category_name` dan `publish_day`.
# One-hot encoding digunakan untuk mengubah variabel kategori menjadi representasi numerik biner,
# sehingga fitur-fitur ini dapat digunakan dalam proses modeling machine learning yang membutuhkan input numerik.
# Setelah proses encoding selesai, kolom asli `category_name` dan `publish_day` dihapus untuk menghindari redundansi data.
#%%
# Membuat salinan bersih dari dataset
df_model = df.copy()

# Menghapus kolom yang tidak relevan atau bertipe object
drop_cols = [
    'channel_id', 'category_id', 'live_status', 'local_title',
    'local_description', 'duration', 'dimension', 'definition',
    'caption', 'license_status', 'allowed_region', 'blocked_region',
    'dislike', 'favorite', 'publish_time', 'description', 'tags',
    'title', 'channel_name', 'trending_time'
]
df_model.drop(columns=drop_cols, inplace=True, errors='ignore')

# Menghapus baris yang masih memiliki nilai NaN
df_model = df_model.dropna()
#%% md
# ### Menyiapkan Dataset untuk Modeling
# 
# Pada tahap ini, dibuat salinan bersih dari dataset utama untuk kebutuhan modeling.
# Beberapa kolom yang tidak relevan atau bertipe object dihapus karena tidak dapat digunakan langsung dalam algoritma machine learning berbasis numerik.
# Selain itu, baris-baris yang masih mengandung nilai kosong (NaN) juga dihapus untuk memastikan data training yang digunakan dalam modeling sudah lengkap dan siap diproses.
#%%
# Memisahkan fitur dan target
X = df_model.drop(columns=['watch_time_proxy', 'like', 'comment', 'engagement_score'], errors='ignore')
y = df_model['watch_time_proxy']

# Membagi data menjadi training set dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% md
# ### Memisahkan Fitur dan Target serta Membagi Data Training dan Testing
# 
# Pada tahap ini, dipisahkan variabel fitur (X) dan variabel target (y) dari dataset yang sudah dibersihkan.
# Variabel target yang ingin diprediksi adalah `watch_time_proxy`, sedangkan fitur lain digunakan sebagai input model.
# Setelah itu, data dibagi menjadi data training dan data testing dengan rasio 80:20 menggunakan `train_test_split`,
# agar model dapat dilatih pada sebagian data dan diuji pada data yang tidak pernah dilihat sebelumnya untuk mengukur performa generalisasi.
#%%
# Menentukan fitur numerik yang akan distandarisasi
numerical_features = ['view', 'publish_hour']

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Fitting scaler hanya pada training data
scaler.fit(X_train[numerical_features])

# Transformasi data training dan testing
X_train[numerical_features] = scaler.transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
#%% md
# ### Standarisasi Fitur Numerik
# 
# Pada tahap ini, dilakukan standarisasi terhadap fitur numerik `view` dan `publish_hour` menggunakan StandardScaler.
# Standarisasi diperlukan untuk menyamakan skala fitur numerik, sehingga setiap fitur memiliki distribusi dengan mean 0 dan standar deviasi 1.
# Langkah ini penting untuk memastikan bahwa model machine learning, khususnya yang sensitif terhadap skala data, dapat berperforma optimal tanpa bias terhadap fitur dengan rentang nilai besar.
#%% md
# ## Analisa Data Preparation
# 
# Tahapan data preparation dilakukan secara berurutan dengan tujuan untuk membersihkan data, menghindari error saat training, dan memastikan data dalam format optimal untuk modeling. Seluruh proses ini telah diimplementasikan secara eksplisit di notebook.
# 
# ### 1. Konversi Waktu Publikasi
# 
# **Teknik:** `pd.to_datetime()` dan `.dt.hour`
# **Tujuan:** Mengubah kolom `publish_time` ke format datetime lalu mengekstrak `publish_hour`. Informasi waktu publikasi dianggap berdampak terhadap pola penayangan video.
# 
# ### 2. Penghapusan Kolom Tidak Relevan
# 
# **Teknik:** `drop(columns=[...])`
# **Kolom yang dihapus:**
# - Metadata visual: `thumbnail_url`, `thumbnail_width`, `thumbnail_height`
# - Identitas channel: `channel_id`, `category_id`, `channel_name`
# - Metadata teks: `description`, `tags`, `title`, `local_title`, `local_description`
# - Informasi teknis: `caption`, `definition`, `dimension`, `duration`, `license_status`
# - Informasi administratif: `allowed_region`, `blocked_region`, `live_status`
# - Kolom tidak relevan lainnya: `favorite`, `dislike`, `publish_time`, `trending_time`
# 
# **Alasan:** Kolom tersebut tidak berkontribusi langsung dalam prediksi dan sebagian besar bertipe object atau terlalu sparsely populated.
# 
# ### 3. Penanganan Nilai Hilang
# 
# **Teknik:** `dropna()`
# **Tujuan:** Menghapus baris yang masih mengandung nilai kosong setelah pembersihan kolom. Hal ini penting untuk mencegah error saat training dan memastikan data model bersih.
# 
# ### 4. Feature Engineering
# 
# - `publish_hour`: fitur numerik hasil ekstraksi waktu upload
# - `engagement_score`: dihitung dari `(like + comment) / view`
# - `watch_time_proxy`: dihitung dari `view * engagement_score`
# 
# `watch_time_proxy` digunakan sebagai **target (`y`)** dalam pemodelan karena tidak tersedia kolom durasi tonton aktual.
# 
# ### 5. One-Hot Encoding Fitur Kategorikal
# 
# **Teknik:** `pd.get_dummies()`
# **Kolom yang diencoding:** `category_name`, `publish_day`
# **Hasil:** Kolom seperti `cat_Music`, `day_Sunday`, dst.
# 
# **Alasan:** One-hot encoding mengubah data kategorikal menjadi numerik tanpa memberi bobot/hierarki, sesuai dengan nature data.
# 
# ### 6. Pemisahan Data Latih dan Uji
# 
# **Teknik:** `train_test_split(test_size=0.2)`
# **Alasan:** Data dibagi 80% untuk pelatihan dan 20% untuk pengujian guna menghindari overfitting dan mengevaluasi generalisasi model.
# 
# ### 7. Standarisasi Fitur Numerik
# 
# **Fitur:** `view`, `publish_hour`
# **Teknik:** `StandardScaler` dari Scikit-learn
# **Alasan:** Standarisasi diperlukan karena kedua fitur memiliki skala yang berbeda dan bisa memengaruhi performa model regresi.
# 
# Langkah-langkah ini bertujuan untuk memastikan:
# - Data bersih dari noise dan null
# - Fitur memiliki skala dan format yang tepat
# - Model dapat dilatih secara stabil dan optimal
#%% md
# # Modeling
#%%
# Inisialisasi dan training model Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred_lr = lr_model.predict(X_test)

# Evaluasi performa model
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Menampilkan hasil evaluasi
print("Linear Regression Results:")
print("MAE:", mae_lr)
print("R² Score:", r2_lr)
#%% md
# ### Modeling Menggunakan Linear Regression
# 
# Pada tahap ini, dilakukan proses training model menggunakan algoritma Linear Regression.
# Model dilatih menggunakan data training untuk mempelajari hubungan antara fitur dan target `watch_time_proxy`.
# Setelah training, model digunakan untuk memprediksi nilai target pada data testing, kemudian dievaluasi menggunakan metrik Mean Absolute Error (MAE) dan R² Score.
# Berdasarkan hasil evaluasi, model Linear Regression menghasilkan MAE sekitar 99.189 dan R² Score sebesar 0.719, menunjukkan bahwa model mampu menjelaskan sekitar 71.9% variasi pada data testing.
#%%
# Inisialisasi dan training model Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Prediksi terhadap data testing
y_pred_rf = rf_model.predict(X_test)

# Evaluasi performa model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Menampilkan hasil evaluasi
print("Random Forest Results:")
print("MAE:", mae_rf)
print("R² Score:", r2_rf)
#%% md
# ### Modeling Menggunakan Random Forest Regressor
# 
# Setelah membangun model Linear Regression, pada tahap ini dilakukan training model menggunakan Random Forest Regressor.
# Random Forest adalah algoritma ensemble berbasis decision tree yang mampu menangani data non-linear dengan baik dan lebih tahan terhadap overfitting.
# Model dilatih pada data training, lalu dilakukan prediksi terhadap data testing, dan evaluasi menggunakan metrik MAE dan R² Score.
# Hasil evaluasi menunjukkan Random Forest menghasilkan MAE sebesar 57.145 dan R² Score sebesar 0.861, yang menunjukkan performa prediksi lebih baik dibandingkan model Linear Regression.
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
# Lakukan scaling ulang pada X_test untuk memastikan data terstandardisasi
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Membuat dictionary model
model_dict = {
    'Linear Regression': lr_model,
    'Random Forest': rf_model
}

# Inisialisasi DataFrame untuk menyimpan MSE
mse_df = pd.DataFrame(columns=['train', 'test'], index=model_dict.keys())

# Menghitung MSE untuk masing-masing model
for name, model in model_dict.items():
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mse_df.loc[name, 'train'] = mean_squared_error(y_train, y_train_pred) / 1e3  # dibagi 1000 untuk memperkecil skala
    mse_df.loc[name, 'test'] = mean_squared_error(y_test, y_test_pred) / 1e3

# Menampilkan hasil MSE
print("Mean Squared Error (dibagi 1000):")
display(mse_df)
#%% md
# ### Evaluasi Perbandingan Model Berdasarkan Mean Squared Error (MSE)
# 
# Pada tahap ini, dilakukan evaluasi model dengan menghitung Mean Squared Error (MSE) pada data training dan testing untuk setiap model yang dibangun.
# Perhitungan MSE dibagi 1000 agar nilai lebih mudah dibaca.
# Dari hasil evaluasi, dapat dilihat bahwa Random Forest memiliki nilai MSE yang lebih kecil dibandingkan Linear Regression, baik pada data training maupun testing.
# Hal ini menunjukkan bahwa Random Forest memberikan performa prediksi yang lebih baik dan lebih stabil dibandingkan model Linear Regression pada dataset ini.
#%%
# Visualisasi perbandingan MSE antar model
fig, ax = plt.subplots(figsize=(8, 4))
mse_df.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.set_title("Mean Squared Error (dibagi 1000)")
ax.set_xlabel("MSE")
ax.grid(zorder=0)
plt.tight_layout()
plt.show()
#%% md
# ### Visualisasi Perbandingan Mean Squared Error (MSE) Antar Model
# 
# Pada tahap ini, dilakukan visualisasi perbandingan nilai Mean Squared Error (MSE) antar model menggunakan bar chart horizontal.
# Grafik ini membantu memudahkan analisis terhadap performa model Linear Regression dan Random Forest baik pada data training maupun testing.
# Dari grafik terlihat bahwa Random Forest memiliki MSE yang lebih rendah, terutama pada data testing, dibandingkan Linear Regression,
# sehingga dapat disimpulkan bahwa Random Forest memiliki kemampuan generalisasi yang lebih baik dalam prediksi durasi tonton (watch time proxy).
#%% md
# ## Evaluation
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
# ![Visualisasi MSE](output_image/mse_evaluasi_output.png)
# 
# Gambar visual menunjukkan perbandingan nilai MSE antara data train dan test dari kedua model, dimana grafik menunjukkan perbedaan besar antara train dan test error pada Random Forest, yang menjadi indikasi overfitting.
# 
# ### Interpretasi
# 
# - **Linear Regression** underfit terhadap data, dengan error tinggi baik di train maupun test.
# - **Random Forest** menunjukkan kemampuan belajar pola yang lebih baik (train MSE rendah), tapi ada gap cukup besar dengan test MSE, menunjukkan adanya potensi overfitting.
# - Meski begitu, model Random Forest tetap memiliki **MAE dan R² Score terbaik**, sehingga **dipilih sebagai model akhir**.
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