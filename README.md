# Laporan Proyek Machine Learning - Junpito Salim    

## Project Overview

Sistem rekomendasi telah menjadi komponen penting dalam meningkatkan pengalaman pengguna di platform streaming, seperti Netflix. Dengan begitu banyaknya konten yang tersedia, pengguna sering kali menghadapi kesulitan dalam menemukan film atau acara TV yang sesuai dengan preferensi mereka. Oleh karena itu, proyek ini bertujuan untuk membangun model sistem rekomendasi yang memanfaatkan kombinasi pendekatan berbasis konten (content-based filtering) dan berbasis pengguna (user-based collaborative filtering) untuk memberikan rekomendasi yang relevan, personal, dan akurat kepada pengguna.

**Pentingnya Proyek**:
1. Meningkatkan Pengalaman Pengguna
Sistem rekomendasi dapat mempersonalisasi pengalaman pengguna dengan menyarankan film atau acara TV yang sesuai dengan minat mereka.

2. Manfaat Bisnis
Sistem rekomendasi yang baik dapat meningkatkan retensi pengguna, waktu tonton, dan kepuasan pelanggan, yang berkontribusi pada keberhasilan bisnis seperti Netflix.
  
 Referensi: 
- Tilak, Garg., Sahil, Shekhar., Prof., Renu, Narwal. (2024). 1. An Evaluation of Machine Learning Algorithms Used for Recommender Systems in Streaming Services. International Journal of Advanced Research in Science, Communication and Technology,  doi: 10.48175/ijarsct-17652
- Fahad, Iqbal, T., R., Gnanajeyaraman. (2023). 2. Hybrid Content-Collaborative Filtering: Improving Collaborative Filtering and Neural Network-based Recommendation.   doi: 10.1109/icses60034.2023.10465386
- Spoorthi, Rakesh. (2023). 3. Movie Recommendation System Using Content Based Filtering. Deleted Journal,  doi: 10.55810/2313-0083.1043
## Business Understanding

### Problem Statements
Pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka. Bagaimana cara merekomendasikan film yang disukai pengguna dengan memanfaatkan metadata dan pola interaksi pengguna?

### Goals
Menghasilkan sistem rekomendasi yang meningkatkan relevansi rekomendasi dengan memadukan dua pendekatan.

### Solution Approach:
* **Content-Based Filtering dengan Cosine Similarity**: Menggunakan metadata film seperti genre, aktor, dan deskripsi.
* **Pendekatan 2:Collaborative filtering menggunakan SVD**: Menggunakan pola rating pengguna untuk merekomendasikan film.


## Data Understanding
Dataset yang digunakan dalam project ini merupakan metadata mocie yang ada di platform netflix yang dirilis di Kaggle, berikut adalah link dataset movie. 
link: [Netflix Movies and TV Shows](hhttps://www.kaggle.com/datasets/shivamb/netflix-shows/data).


Variabel-variabel pada Netflix Movies and TV Shows adalah sebagai berikut:
 -   `show_id`: Identifikasi unik untuk setiap entri pada dataset. Berfungsi sebagai primary key.     
 -   `type`: Menunjukkan jenis konten, seperti Movie atau TV Show.         
 -   `title`: Judul dari konten (film atau acara TV).        
 -   `director`: Nama sutradara dari konten tersebut.     
 -   `cast`: Nama-nama pemeran dalam konten.         
 -   `country`: Negara asal konten.      
 -   `date_added`: Tanggal ketika konten ditambahkan ke platform.   
 -   `release_year`: Tahun rilis asli konten. 
 -   `rating`: Kategori umur atau klasifikasi konten (misalnya, PG, R, TV-MA).       
 -   `duration`: Durasi konten (misalnya, dalam menit untuk film atau jumlah episode untuk acara TV).     
 -  `listed_in`: Genre atau kategori di mana konten terdaftar (misalnya, Drama, Comedy).    
 -  `description`: Deskripsi singkat tentang konten.  


Selanjutnya data rating User dari neflik juga digunakan pada project ini untuk pendekatan colaborative filtering. berikut adalah link dataset.
link: [netflix prize data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data?select=combined_data_1.txt)

Variable- variable pada netflix prize data adalah sebagai berikut:
- `Cust_Id`: Identifikasi unik untuk setiap customer dalam dataset.
- `Rating`: Nilai yang diberikan oleh user pada movie atau TV show.
- `Movie_Id`: Identifikasi unik untuk setiap movie atau TV show dalam dataset.

**Ekplorasi Data Analisis**:

Pada tahap ini, dilakukan eksplorasi terhadap data untuk memahami distribusi, pola, serta hubungan antar variabel.

**Netflix Movies and TV Shows**
1. **Meninjau Struktur Dataset:** Pada langkah ini kita akan melihat jumlah kolom, type data dan nilai yang terdapat pada dataset. Berikut adalah rangkuman informasi dari proses ini:
    * Dataset memiliki total 8807 entri (baris) dan 12 kolom.
    * Setiap entri dalam dataset mewakili satu item atau acara (show) dengan atribut yang mendeskripsikan informasi terkait acara tersebut.
    * Object (String): Kolom dengan tipe data objek terdiri dari string atau teks, meliputi kolom seperti `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `rating`, `duration`, `listed_in`, dan `description`.
    * Kolom `release_year` berisi tahun rilis acara dengan tipe data bilangan bulat (int64).
    * Setelah diperiksa terdapat beberapa variable yang memiliki nilai kosong, seperti director,cast dan country. setelah data null dibersihkan jumlah data yang tersisa adalah: 5332 baris.

2.  **Visualisasi:** Pada tahap ini Data divisualisasikan untuk mendapatkan informasi lebih lanjut dari data, berikut adalah informasi dari hasil visualisasi:
    *  Jumlah Movie pada dataset jauh lebih banyak daripada tv show
    *  Dari tahun rilis dapat dilihat bahwa jumlah movie meningkat pesat pada tahun 2000 ke atas, dan masih meningkat pesat di 2020an.
    *  Rating dengan jumlah terbanyak pada dataset merupakan TV-MA diikuti TV-14 dan R.
    *  Genre film paling banyak adalah International movie diikuti Dramas dan Comedies
    *  Negara asal content terbanyak adalah United Stated kemudian India dan United kingdom

**netflix prize data**

 
1.   Struktur Dataset:
        * Dataset terdiri dari 24.053.76444 total rating yang diberikan oleh user.
        * Total user yang memberikan rating adalah 4470758 user, kedalam 4.499 Movie.
        * Terdapat 3884 movie dan 79 user dengan jumlah review yang sedikit, sehingga data tersebut dihapus untuk mengurangi dimensi.
2. Insight dari visualisasi:
    * Rating paling banyak diberikan user adalah rating 4 dengan total 34% dari keseluruhan user diikuti oleh rating 3 dengan total 29% user dan ratig 5 dengan total 23% user. 
 

## Data Preparation
Proses data preparation dilakukan untuk memastikan data yang digunakan bersih, relevan, dan dapat digunakan dalam analisis atau model. 

**Penjelasan tahapan data preparation untuk content base filtering**: 
 1. Penghapusan Nilai Kosong atau Hanya Berisi Spasi:
    -  Baris yang memiliki nilai kosong atau hanya berisi spasi dihapus untuk memastikan kualitas data.
    -  Menghilangkan baris dengan data tidak valid menghindari bias atau kesalahan saat analisis.
 2. Kolom yang tidak relevan untuk analisis dihapus, menyisakan hanya kolom berikut: `title` `director` `cast` `listed_in` `description`.
 3. Text Cleaning: Dilakukan pembersihan teks pada kolom description untuk memastikan teks bebas dari karakter yang tidak relevan (seperti simbol atau tanda baca berlebih) sebelum digunakan lebih lanjut.
 4. Ekstraksi Kata Kunci dari Kolom Deskripsi
Ekstraksi kata kunci dilakukan menggunakan algoritma RAKE (Rapid Automatic Keyword Extraction) untuk menangkap informasi utama dari deskripsi. Kata kunci disimpan dalam kolom baru bernama Key_words. Setelah proses ini selesai, kolom description dihapus karena sudah tidak diperlukan.
5. Ekstraksi Fitur dengan CountVectorizer: Kolom `Key_words`, `director`, `cast`, dan `listed_in` digabungkan menjadi satu kolom baru bernama `bag_of_words`. Proses ini menyederhanakan representasi fitur untuk model berbasis teks. CountVectorizer digunakan untuk menghasilkan matriks jumlah kata (count matrix) dari kolom `bag_of_words`.


**Penjelasan untuk tahapan data preparation untuk collaborative filtering** 
1. Conversi data: Kolom Rating dikonversi menjadi tipe data float untuk memastikan dapat digunakan dalam analisis lebih lanjut.
2. Penanganan Missing Values : Missing values dalam kolom Rating diidentifikasi dan digunakan untuk membagi data menjadi segmen film berdasarkan indeks.
3. Seleksi Data: statistik ulasan dihitung untuk setiap film dan pelanggan. Film dan pelanggan yang jumlah ulasannya di bawah kuantil 80% dihapus untuk mengurangi noise.
4. Pembuatan Pivot Table: data yang telah dibersihkan diubah menjadi format pivot table untuk menghasilkan matriks pelanggan-film yang akan digunakan dalam Collaborative Filtering.

## Modeling
Pada bagian ini, saya mengembangkan sistem rekomendasi menggunakan dua pendekatan algoritma berbeda untuk menyelesaikan permasalahan, yaitu Content-Based Filtering dengan Cosine Similarity dan Collaborative Filtering menggunakan Singular Value Decomposition (SVD).

**Pendekatan 1: Content-Based Filtering dengan Cosine Similarity** 

Cosine similarity adalah metrik yang digunakan untuk mengukur kesamaan antara dua vektor berdasarkan sudut kosinus di antara mereka. Nilainya berkisar antara 0 hingga 1, di mana:

- 0 berarti vektor tidak memiliki kesamaan.
- 1 berarti vektor identik.

Dalam konteks ini, representasi numerik film (dari CountVectorizer) dibandingkan untuk menentukan tingkat kemiripan konten antarfilm. Rumusnya adalah:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$

di mana:

- \( A \) dan \( B \) adalah vektor teks dua film.
- \( A \cdot B \) adalah hasil perkalian dot product.
- \( \|A\| \) dan \( \|B\| \) adalah panjang (magnitude) masing-masing vektor.

Pendekatan ini memanfaatkan CountVectorizer untuk mengubah fitur teks dari data (kolom bag_of_words) menjadi representasi numerik (count matrix). Kesamaan antar film dihitung menggunakan matriks kesamaan kosinus (cosine similarity matrix), yang mengukur kemiripan antara vektor representasi film.
- Kelebihan dan kekurangan:
    - Kelebihan:
        - Mudah diimplementasikan.
        - Tidak memerlukan data pengguna atau riwayat interaksi (cold-start problem).
        - Efektif untuk data dengan fitur teks kaya informasi.

    - Kekurangan:
        - Hanya merekomendasikan film dengan kemiripan konten (tidak menangkap preferensi pengguna).
        - Performanya dapat menurun pada dataset besar karena hitungan kesamaan kosinus berbasis matriks yang besar.
    
- Top 10 hasil rekomendasi untuk judul film "Adrift" :
    - 'The 12th Man',
    - 'Needhi Singh',
     - 'Loving',
    - 'Paskal',
    - 'Defiance',
    - 'The Mirror Boy',
    - 'Shootout at Lokhandwala',
    - 'The Siege of Jadotville',
    - 'Shiva',
    - 'Mosul'

**Pendekatan 2:Collaborative filtering menggunakan SVD**
Collaborative Filtering berbasis Singular Value Decomposition (SVD) adalah pendekatan matriks untuk menghasilkan rekomendasi berdasarkan data interaksi pengguna terhadap film, seperti rating. Teknik ini memanfaatkan dekomposisi matriks untuk menemukan pola laten yang menghubungkan pengguna dengan item (film), tanpa memerlukan informasi tambahan dari konten film.
berikut adalah rumus SVD:
        $$\mathbf{R}_{m \times n} = \mathbf{U}_{m \times k} \cdot \boldsymbol{\Sigma}_{k \times k} \cdot \mathbf{V}^T_{k \times n}$$
di mana:
    - R (m x n): Matriks R berukuran m x n
    - U (m x k): Matriks U berukuran m x k
    - Σ (k x k): Matriks diagonal berukuran k x k
    - V^T (k x n): Matriks V berukuran k x n

- Dalam penerapannya Model SVD diinisialisasi menggunakan library `Surprise`
- Kelebihan dan kekurangan:
    - Kelebihan : 
        - Mampu menangkap pola laten antara pengguna dan item, memberikan rekomendasi yang lebih personal.
        - Tidak memerlukan data tambahan seperti fitur konten film (hanya berdasarkan data interaksi).
    - Kekurangan: 
    
        - Membutuhkan data interaksi yang cukup banyak untuk memberikan hasil optimal (masalah cold-start untuk pengguna baru).
        - Kompleksitas perhitungan meningkat untuk dataset yang sangat besar.
- Top 10 rekomendasi User user_id=78531 :
    - Immortal Beloved  
    -    Lilo and Stitch  
    - James and the Giant Peach  
    - A Good Marriage  
    - Sade: Sade Live  
    - Tall in the Saddle  
    - All Over Me 
    - Out of the Past  
    - Tragic Hero 
    - NYPD Blue: Season 1 
        


## Evaluation
-**precision**  :
Precision digunakan untuk mengukur seberapa banyak rekomendasi yang relevan dibandingkan dengan jumlah total rekomendasi yang diberikan. Dalam konteks Content-Based Filtering, precision dihitung dengan membandingkan item yang relevan berdasarkan metadata film (seperti genre, aktor, deskripsi) dengan total rekomendasi yang dihasilkan.
- formula : $$P = \frac{\text{Jumlah item relevan}}{\text{Jumlah rekomendasi}}$$

Hasil evaluasi untuk Content-Based Filtering menunjukkan bahwa model ini berhasil menghasilkan precision sebesar 90%. Artinya, 90% dari rekomendasi yang diberikan adalah relevan bagi pengguna.

**Dampak terhadap Problem Statement**:
Precision yang tinggi menunjukkan bahwa sistem rekomendasi berbasis konten dapat memberikan film yang sesuai dengan preferensi pengguna berdasarkan metadata film. Ini menjawab problem statement yang menyatakan bahwa pengguna kesulitan menemukan film yang sesuai dengan preferensi mereka.

**Dampak terhadap Goals:**
Dengan precision 90%, model ini berhasil meningkatkan relevansi rekomendasi film kepada pengguna, yang sejalan dengan tujuan utama untuk menghasilkan sistem rekomendasi yang lebih tepat sasaran.

**Root mean Squared Error(RMSE)**
RMSE digunakan untuk mengevaluasi model Collaborative Filtering, yang berfokus pada pola interaksi dan rating pengguna untuk merekomendasikan film. RMSE mengukur rata-rata kesalahan prediksi model, dengan memberikan penalti lebih besar pada kesalahan prediksi yang besar.
rumus RMSE adalah sebagai berikut:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

Keterangan:
- y_i: Rating sebenarnya.
- ŷ_i: Rating yang diprediksi.
- n: Jumlah sampel.

**Hasil evaluasi dengan RMSE pada Collaborative Filtering** menunjukkan nilai RMSE  (0.9810 Fold 1) (0.9818 Fold 2) (0.9805 Fold 3) (0.9891 Fold 4) (0.9944 Fold 5) dengan rata-rata **0.9853**. Nilai RMSE yang rendah ini menunjukkan bahwa model mampu memprediksi rating pengguna dengan cukup akurat, yang berarti rekomendasi yang diberikan berdasarkan pola rating pengguna sangat relevan.

**Dampak terhadap Problem Statement:**
RMSE yang rendah menunjukkan bahwa model collaborative filtering dapat memberikan rekomendasi yang sangat mendekati rating yang diinginkan oleh pengguna. Hal ini membantu pengguna dalam menemukan film yang mereka sukai berdasarkan pola interaksi dan rating yang telah diberikan.

**Dampak terhadap Goals:**
Dengan menggunakan pendekatan Collaborative Filtering dan mengurangi kesalahan prediksi, sistem ini berhasil mencapai tujuan untuk meningkatkan relevansi rekomendasi melalui pemanfaatan pola rating pengguna.


**---Ini adalah bagian akhir laporan---**


