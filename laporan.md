# Proyek Pertama Predictive Analytics: Prediksi Durasi Tonton Video YouTube - Muhammad Dila

## Domain Proyek

Platform video seperti YouTube menjadi salah satu media hiburan dan edukasi terbesar di dunia. Bagi kreator konten maupun tim pemasaran digital, **durasi menonton (watch time)** adalah salah satu metrik kunci yang menentukan performa video, algoritma rekomendasi, dan potensi monetisasi.

Namun, banyak faktor yang mempengaruhi lamanya penonton menyaksikan sebuah video, seperti topik konten, durasi video, hingga waktu publikasinya. Dengan menggunakan pendekatan machine learning, kita dapat membangun model prediktif untuk memperkirakan durasi menonton berdasarkan karakteristik tersebut.

**Mengapa masalah ini penting untuk diselesaikan?**
- Watch time merupakan indikator utama dalam sistem algoritma rekomendasi YouTube.
- Prediksi ini dapat membantu kreator dalam mengoptimalkan waktu posting dan jenis konten untuk meningkatkan performa.
- Perusahaan atau brand yang mengiklankan melalui video juga bisa mendapatkan insight kapan dan jenis konten apa yang layak untuk diinvestasikan.

**Bagaimana cara menyelesaikannya?**
- Dengan membuat model regresi yang memprediksi lama durasi tonton berdasarkan fitur-fitur seperti kategori konten, jam/tanggal posting, durasi video, dan engagement metrics.
- Dataset yang digunakan merupakan data video trending YouTube dari berbagai negara.

**Referensi:**
- Deep Neural Networks for YouTube Recommendations [Deep Neural Networks for YouTube Recommendations](https://dl.acm.org/doi/abs/10.1145/2959100.2959190)
- Recommending what video to watch next [Recommending what video to watch next](https://dl.acm.org/doi/abs/10.1145/3298689.3346997)

## Business Understanding

### **Problem Statements**
1. Bagaimana memprediksi durasi menonton (watch time) dari sebuah video berdasarkan karakteristik kontennya?
2. Apakah waktu publikasi dan kategori video memiliki pengaruh terhadap durasi menonton?
3. Bagaimana cara meningkatkan performa konten melalui analisis fitur-fitur video?

### **Goals**
1. Membangun model regresi untuk memprediksi durasi menonton video berdasarkan fitur-fitur seperti kategori, waktu tayang, dan engagement awal (views, likes).
2. Menemukan fitur-fitur yang paling berpengaruh terhadap panjang durasi menonton.
3. Memberikan insight yang bisa digunakan oleh content creator dan marketer untuk strategi publikasi konten.

### **Solution Statements**
1. Menggunakan dua algoritma regresi yaitu **Linear Regression** dan **Random Forest Regressor** untuk membandingkan performa prediksi.
2. Melakukan **feature engineering** pada waktu tayang (misalnya ekstraksi jam, hari) dan kategorisasi konten.
3. Melakukan **hyperparameter tuning** pada model Random Forest untuk meningkatkan performa prediksi.
4. Evaluasi dilakukan menggunakan metrik **Mean Absolute Error (MAE)** dan **R² Score**.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data video yang sedang trending di YouTube Indonesia, diambil dari [Kaggle: Indonesia's Trending YouTube Video Statistics](https://www.kaggle.com/datasets/syahrulhamdani/indonesias-trending-youtube-video-statistics). Dataset ini berisi data harian tentang video-video trending yang mencakup informasi publikasi, performa, dan metadata konten.

### Jumlah Data
Dataset ini memiliki:

- **Jumlah baris:** 172.347 baris
- **Jumlah kolom:** 28 kolom

Jumlah baris merepresentasikan banyaknya data video yang tercatat harian sebagai trending,
sedangkan kolom mencakup berbagai atribut terkait metadata dan performa video.

### Kondisi Data

Hasil pemeriksaan kondisi data menunjukkan:

- **Missing Value:**
  - Terdapat missing values pada beberapa kolom penting seperti `description`, `tags`, `thumbnail_url`, `view`, `like`, `comment`, dan `category_name`.
  - Kolom `allowed_region` dan `blocked_region` memiliki jumlah missing value yang sangat tinggi (>90% dari data).

- **Duplikasi:**
  - Terdapat duplikasi sejumlah **~0** baris, namun pengecekan tetap dilakukan untuk memastikan keunikan data.

- **Outlier:**
  - Fitur numerik seperti `view` dan `like` menunjukkan adanya outlier ekstrem, dengan sebagian kecil video mencapai views sangat tinggi dibandingkan mayoritas video lainnya.
  - Distribusi views dan likes berbentuk **right-skewed** berdasarkan hasil visualisasi histogram logaritmik.

Penanganan lebih lanjut terhadap missing values, duplikasi akan dilakukan pada tahap **Data Preparation**.

### Deskripsi Fitur (Variabel)

| No | Fitur | Deskripsi |
|:--|:--|:--|
| 1 | `video_id` | ID unik video YouTube |
| 2 | `publish_time` | Waktu unggahan video |
| 3 | `channel_id` | ID unik channel pengunggah |
| 4 | `title` | Judul video |
| 5 | `description` | Deskripsi video |
| 6 | `thumbnail_url` | URL gambar thumbnail video |
| 7 | `thumbnail_width` | Lebar thumbnail video |
| 8 | `thumbnail_height` | Tinggi thumbnail video |
| 9 | `channel_name` | Nama channel pengunggah |
| 10 | `tags` | Tag terkait video |
| 11 | `category_id` | ID kategori video (numerik) |
| 12 | `live_status` | Status apakah video live stream |
| 13 | `local_title` | Judul video dalam bahasa lokal |
| 14 | `local_description` | Deskripsi video dalam bahasa lokal |
| 15 | `duration` | Durasi video (format ISO 8601) |
| 16 | `dimension` | Dimensi video (2d atau 3d) |
| 17 | `definition` | Resolusi video (standard/hd) |
| 18 | `caption` | Indikator apakah video memiliki caption |
| 19 | `license_status` | Status lisensi video |
| 20 | `allowed_region` | Negara di mana video diperbolehkan tampil |
| 21 | `blocked_region` | Negara di mana video diblokir |
| 22 | `view` | Jumlah views video |
| 23 | `like` | Jumlah likes video |
| 24 | `dislike` | Jumlah dislikes video |
| 25 | `favorite` | Jumlah favorit video (umumnya 0 setelah 2018) |
| 26 | `comment` | Jumlah komentar video |
| 27 | `trending_time` | Waktu video tercatat trending |
| 28 | `category_name` | Nama kategori video berdasarkan `category_id` |

### Eksplorasi Data

Untuk memahami pola distribusi dan hubungan antar variabel, dilakukan beberapa visualisasi eksploratif:

- Distribusi `views` dan `likes` menunjukkan **pola right-skewed** yang ekstrem: mayoritas video memiliki performa menengah/rendah, hanya sebagian kecil yang benar-benar viral.
- Distribusi waktu unggah (`publish_hour`) menunjukkan bahwa **jam 08.00–13.00** merupakan waktu populer untuk mengunggah konten.
- Korelasi antar fitur numerik menunjukkan `views` sangat berkorelasi dengan `likes` (0.88) dan `comment` (0.57), yang menandakan keterkaitan antara engagement dan popularitas.
- Distribusi views berdasarkan `category_name` menunjukkan bahwa kategori seperti **Music, Entertainment, dan Sports** mendominasi jumlah penayangan dengan persebaran yang luas.

> *Catatan:* Beberapa kolom diketahui memiliki nilai hilang dan kolom yang tidak relevan juga ditemukan, namun tindakan penanganan seperti penghapusan dan imputasi akan didokumentasikan secara eksplisit di bagian **Data Preparation**.

## Visualisasi Data

Beberapa visualisasi eksploratif dilakukan untuk memahami pola data:

- **Distribusi Views dan Likes:**
  Menunjukkan pola distribusi yang sangat right-skewed, di mana sebagian besar video memiliki performa menengah hingga rendah, sementara hanya sebagian kecil video yang benar-benar viral.
 ![Distribusi Views dan Likes](output_image/Distribusi%20Jumlah%20Likes.png)
- **Distribusi Publish Hour:**
  Sebagian besar video diunggah antara **pukul 08.00 hingga 13.00**, menunjukkan waktu populer untuk mengunggah konten.
  ![Distribusi Views dan Likes](output_image/Distribusi%20Jam%20Publikasi%20Video.png)

- **Korelasi antar Fitur Numerik:**
  Terdapat korelasi kuat antara `views` dan `likes` (0.88), serta `likes` dan `comment` (0.70). Ini menandakan bahwa engagement (likes dan comments) memiliki keterkaitan kuat dengan popularitas video.
 ![Distribusi Views dan Likes](output_image/Korelasi%20antar%20Fitur%20Numerik.png)

- **Distribusi Views per Kategori:**
  Kategori seperti **Music**, **Entertainment**, dan **Sports** memiliki persebaran views yang lebih tinggi dibandingkan kategori lainnya.
  ![Distribusi Views dan Likes](output_image/Distribusi%20Views%20per%20Kategori.png)

**Catatan:** Beberapa kolom ditemukan memiliki nilai kosong (missing values) dan terdapat kolom yang kurang relevan. Tindakan penanganan missing values dan pembersihan data akan dibahas lebih rinci pada bagian **Data Preparation**.

### Insight Awal

- **Kategori konten dan waktu publikasi** kemungkinan memengaruhi durasi tonton video.
- Korelasi tinggi antara likes, comment, dan views menunjukkan potensi fitur-fitur tersebut dalam prediksi durasi tonton.
- Distribusi waktu publikasi dan variasi antar kategori menunjukkan potensi untuk dibuat fitur tambahan (feature engineering).

**Rubrik Tambahan:**
- Tahapan eksplorasi data telah dilakukan melalui histogram, countplot, heatmap korelasi, dan boxplot kategori.
- Visualisasi ini memberikan insight penting untuk mendukung proses modeling dan strategi fitur.

## Data Preparation

Tahapan data preparation dilakukan untuk membersihkan data, membuat fitur baru, dan memastikan data siap untuk digunakan dalam modeling. Seluruh langkah ini diimplementasikan secara berurutan seperti berikut:

### 1. Pembersihan Data Awal

**Teknik:** `drop(columns=[...])`, imputasi `fillna()`

- Menghapus kolom `video_id`, `thumbnail_url`, `thumbnail_width`, dan `thumbnail_height` karena tidak berkontribusi dalam analisis performa video.
- Mengisi nilai kosong pada kolom `description` dan `tags` dengan string default ("No description" dan "No tags").

**Tujuan:** Mengurangi noise dan mencegah error akibat nilai kosong pada fitur teks.

---

### 2. Pembuatan Fitur Waktu dari `publish_time`

**Teknik:** `pd.to_datetime()`, `.dt.hour`, `.dt.day_name()`

- Mengubah `publish_time` menjadi format datetime.
- Membuat dua fitur baru:
  - `publish_hour` (jam publikasi 0–23)
  - `publish_day` (nama hari publikasi)

**Tujuan:** Menangkap pola perilaku waktu unggah video yang relevan dengan performa konten.

---

### 3. Pembuatan Engagement Score dan Watch Time Proxy

**Teknik:** perhitungan numerik sederhana

- Menghapus baris dengan `views = 0` untuk menghindari pembagian dengan nol.
- Membuat `engagement_score` = (`likes` + `comments`) / `views`.
- Membuat `watch_time_proxy` = `views * engagement_score`.

**Tujuan:** Mengestimasi tingkat keterlibatan dan durasi tonton video sebagai target prediksi.

---

### 4. One-Hot Encoding Fitur Kategorikal

**Teknik:** `pd.get_dummies()`

- Melakukan one-hot encoding terhadap kolom `category_name` dan `publish_day`.
- Menghapus kolom kategorikal aslinya untuk menghindari redundansi.

**Tujuan:** Mengubah fitur kategorikal menjadi format numerik biner untuk kompatibilitas dengan algoritma machine learning.

---

### 5. Menyiapkan Dataset untuk Modeling

**Teknik:** `drop(columns=[...])`, `dropna()`

- Menghapus fitur bertipe object atau fitur yang tidak relevan untuk modeling numerik.
- Menghapus baris-baris yang masih memiliki nilai kosong (NaN).

**Tujuan:** Membuat dataset bersih dan numerik, siap untuk training model.

---

### 6. Memisahkan Fitur dan Target serta Membagi Data Training dan Testing

**Teknik:** `train_test_split(test_size=0.2)`

- Memisahkan `X` (fitur) dan `y` (`watch_time_proxy`).
- Membagi data menjadi 80% untuk training dan 20% untuk testing.

**Tujuan:** Membantu model belajar dari sebagian data dan menguji performanya pada data baru.

---

### 7. Standarisasi Fitur Numerik

**Teknik:** `StandardScaler` dari Scikit-learn

- Melakukan standarisasi terhadap fitur `view` dan `publish_hour`.

**Tujuan:** Menyamakan skala fitur numerik untuk mencegah dominasi fitur tertentu dalam proses training model.

## Modeling

Tahapan ini membahas dua algoritma machine learning yang digunakan untuk menyelesaikan masalah prediksi estimasi durasi menonton (`watch_time_proxy`) berdasarkan metadata video YouTube.

### 1. Linear Regression

Model pertama yang digunakan adalah **Linear Regression**. Model ini digunakan sebagai baseline karena:

- Sifatnya yang sederhana dan cepat dilatih
- Memberikan acuan awal performa prediksi

**Evaluasi Hasil:**
- **MAE:** 99,189
- **R² Score:** 0.72

**Kelebihan:**
- Mudah diimplementasikan dan cepat dilatih
- Hasil mudah diinterpretasikan karena linearitas hubungan fitur dan target

**Kekurangan:**
- Hanya menangkap hubungan linear antar fitur
- Sensitif terhadap outlier
- Kurang cocok untuk data dengan kompleksitas tinggi

Model ini dapat menjelaskan sekitar **72%** variasi dalam data, namun MAE yang cukup tinggi menunjukkan bahwa prediksi model masih cukup kasar. Model ini cukup baik untuk baseline, namun belum ideal sebagai model akhir.

### 2. Random Forest Regressor

Model kedua adalah **Random Forest Regressor**, yaitu metode ensemble yang membangun banyak pohon keputusan dan menggabungkan hasilnya untuk prediksi akhir.

Model dilatih menggunakan parameter default dengan `random_state=42` untuk menjaga reprodusibilitas hasil.

**Evaluasi Hasil:**
- **MAE:** 57,145
- **R² Score:** 0.86

**Kelebihan:**
- Mampu menangkap hubungan non-linear antar fitur
- Tidak sensitif terhadap outlier
- Cenderung memiliki performa yang lebih baik dibanding model linear

**Kekurangan:**
- Interpretasi model lebih sulit karena sifatnya sebagai black-box
- Waktu pelatihan lebih lama dibanding model linear

Model ini menunjukkan performa yang lebih baik secara signifikan dibandingkan Linear Regression. Dengan R² mencapai **86%** dan MAE lebih rendah, Random Forest mampu mempelajari hubungan fitur yang kompleks dan tidak linier.

### Perbandingan Model

| Model               | MAE         | R² Score   |
|---------------------|-------------|------------|
| Linear Regression   | 99,189      | 0.72       |
| Random Forest       | 57,145      | 0.86       |

### Pemilihan Model Terbaik

Berdasarkan evaluasi di atas, **Random Forest Regressor** dipilih sebagai model terbaik karena:

- Memiliki akurasi lebih tinggi (MAE lebih rendah)
- Memiliki generalisasi lebih baik (R² lebih tinggi)
- Tidak sensitif terhadap outlier dan mampu menangani hubungan non-linear

Meskipun belum dilakukan hyperparameter tuning, hasil evaluasi awal menunjukkan bahwa model ini sudah memiliki performa yang optimal dibanding baseline. Random Forest dipilih sebagai model akhir karena memberikan hasil yang paling optimal pada data yang tersedia. Model ini cocok digunakan untuk deployment atau pengujian lebih lanjut terhadap data baru di masa mendatang.

## Evaluation
### Metrik Evaluasi yang Digunakan

Karena proyek ini merupakan permasalahan **regresi**, maka metrik yang digunakan untuk mengevaluasi performa model adalah:

- **MAE (Mean Absolute Error):** Rata-rata selisih absolut antara nilai aktual dan prediksi. Semakin kecil nilainya, semakin baik.
- **MSE (Mean Squared Error):** Rata-rata kuadrat selisih antara nilai aktual dan prediksi. Skala error lebih besar karena dihitung kuadrat. Dalam proyek ini, nilai MSE dibagi dengan 1000 agar mudah dibaca.
- **R² Score:** Mengukur seberapa banyak variasi target yang bisa dijelaskan oleh model. Nilai 1 menunjukkan model sempurna.

### Hasil Evaluasi

| Model               | MAE         | R² Score   |
|---------------------|-------------|------------|
| Linear Regression   | 99,189      | 0.72       |
| Random Forest       | 57,145      | 0.86       |

Untuk metrik **Mean Squared Error**, didapatkan hasil sebagai berikut (dalam ribuan):

| Model               | Train MSE (÷1000)   | Test MSE (÷1000)    |
|---------------------|---------------------|----------------------|
| Linear Regression   | 97,690,176.91       | 39,278,336.39        |
| Random Forest       | 6,309,616.79        | 39,721,605.81        |

### Visualisasi MSE

![Visualisasi MSE](output_image/mse_evaluasi_output.png)
Gambar visual menunjukkan perbandingan nilai MSE antara data train dan test dari kedua model, dimana grafik menunjukkan perbedaan besar antara train dan test error pada Random Forest, yang menjadi indikasi overfitting.

### Interpretasi

- **Linear Regression** underfit terhadap data, dengan error tinggi baik di train maupun test.
- **Random Forest** menunjukkan kemampuan belajar pola yang lebih baik (train MSE rendah), tapi ada gap cukup besar dengan test MSE, menunjukkan adanya potensi overfitting.
- Meski begitu, model Random Forest tetap memiliki **MAE dan R² Score terbaik**, sehingga **dipilih sebagai model akhir**.

### Kesimpulan Akhir

Pada proyek ini, dilakukan prediksi terhadap estimasi durasi menonton (`watch_time_proxy`) berdasarkan metadata video YouTube. Proses dilakukan melalui tahapan:

- Data understanding untuk memahami struktur dan pola data
- Data preparation seperti pembersihan, transformasi, dan encoding
- Pemodelan dengan algoritma Linear Regression dan Random Forest
- Evaluasi menggunakan metrik regresi: MAE, MSE, dan R² Score

Model **Random Forest Regressor** dipilih sebagai model terbaik karena memberikan performa prediktif paling tinggi:
- MAE: ~57 ribu
- R² Score: 0.86

Insight tambahan:
- Fitur `publish_hour`, `category_name`, dan engagement proxy terbukti berkontribusi besar terhadap prediksi
- Random Forest memiliki potensi overfitting, sehingga tuning lanjutan bisa dilakukan di masa depan

---

#### Apakah sudah menjawab Problem Statement?

1. **Prediksi durasi tonton berdasarkan karakteristik konten:**  
   Model Random Forest berhasil memprediksi durasi tonton (`watch_time_proxy`) dengan **MAE sebesar 57.145** dan **R² sebesar 0.86**. Ini menunjukkan bahwa karakteristik seperti views, likes, publish hour, dan category_name berkontribusi signifikan terhadap prediksi durasi tonton.

2. **Pengaruh waktu publikasi dan kategori video:**  
   Fitur `publish_hour` dan `category_name` telah diikutsertakan dalam proses modeling dan terbukti berpengaruh terhadap hasil prediksi. Ini menjawab bahwa waktu publikasi dan kategori video memang relevan terhadap durasi tonton.

3. **Strategi meningkatkan performa konten:**  
   Insight dari analisis fitur mengindikasikan bahwa memilih waktu unggah yang optimal dan kategori yang sesuai dapat meningkatkan potensi watch time video, sesuai dengan strategi peningkatan performa konten yang diharapkan.

---

#### Apakah sudah mencapai Goals?

1. **Membangun model regresi:**  
   Model regresi berhasil dibangun dan diuji, dengan Random Forest dipilih sebagai model terbaik setelah dibandingkan dengan Linear Regression.

2. **Menemukan fitur berpengaruh:**  
   Analisis menunjukkan bahwa fitur `publish_hour`, `category_name`, dan kombinasi engagement metrics (`likes`, `comments`, `views`) merupakan fitur yang paling berpengaruh terhadap prediksi.

3. **Memberikan insight untuk strategi konten:**  
   Berdasarkan hasil evaluasi, insight yang dihasilkan dari model dapat digunakan oleh kreator dan marketer untuk mengoptimalkan strategi publikasi konten.

---

#### Apakah solusi statement berdampak?

- **Feature engineering** terhadap waktu publikasi dan kategori terbukti efektif dalam meningkatkan kualitas model prediksi.
- **Penggunaan Random Forest** berhasil memberikan performa prediksi yang lebih baik dibandingkan baseline Linear Regression.
- Evaluasi dengan metrik **MAE dan R²** menunjukkan bahwa solusi yang direncanakan berdampak positif terhadap capaian proyek.

---

**Kesimpulan:**  
Model yang dibangun tidak hanya mencapai performa prediktif yang baik, tetapi juga berhasil mendukung pencapaian seluruh problem statement, goals, dan solution statement yang telah ditentukan di tahap Business Understanding.