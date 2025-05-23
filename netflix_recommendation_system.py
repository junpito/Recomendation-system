# -*- coding: utf-8 -*-
"""netflix-recommendation-system.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qxXN_T0lWTqSs0al-u3H8TKiaEl4GxAZ

# project akhir: Movie Recomendation System

Nama: Junpito Salim

Id Dicoding:

Sistem rekomendasi telah menjadi komponen penting dalam meningkatkan pengalaman pengguna di platform streaming, seperti Netflix. Dengan begitu banyaknya konten yang tersedia, pengguna sering kali menghadapi kesulitan dalam menemukan film atau acara TV yang sesuai dengan preferensi mereka. Oleh karena itu, proyek ini bertujuan untuk membangun model sistem rekomendasi yang memanfaatkan kombinasi pendekatan berbasis konten (content-based filtering) dan berbasis pengguna (user-based collaborative filtering) untuk memberikan rekomendasi yang relevan, personal, dan akurat kepada pengguna.

# 1| Persiapan

## 1.1| Persiapan library yang dibutuhkan
"""

pip install rake_nltk

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import linear_kernel

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

"""## 1.2| Mempersiapkan dataset

Dataset diambil untuk proses pembuatan model menggunakan file `netflix_titles.csv`. Data dimuat ke dalam DataFrame dan ditampilkan beberapa baris awal untuk eksplorasi awal struktur dan isi data.
"""

#ambil dataset untuk conten base filtering
df_movie = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')
df_movie.head()

df_movie.tail()

"""# 2 | Data Understanding and Cleaning data"""

df_movie.shape

"""kita mendapatkan ukuran dataset, yaitu 8807 baris dan 12 kolom."""

df_movie.info()

"""- Terdapat 11 kolom bertipe objek (string), termasuk informasi seperti `title`, `director`, dan `cast`.
-
Satu kolom bertipe numerik:` release_yea`r (tipe int64).

Perintah `df_movie.isnull().sum()` menghitung jumlah nilai kosong (missing values) di setiap kolom. Langkah ini penting untuk mendeteksi kolom yang mungkin membutuhkan pembersihan data lebih lanjut, misalnya dengan mengisi nilai yang hilang atau menghapus data yang tidak relevan.
"""

df_movie.isnull().sum()

"""Terdapat Nilai kosong pada beberapa variabel seperti director,cast dan country

Nilai kosong pada variabel ini tidak diimputasi karena data yang hilang dapat menyebabkan bias. Baris dengan nilai kosong pada variabel ini akan dihapus agar hasil rekomendasi lebih akurat.
"""

#membersihkan data null
cleaned_df= df_movie.dropna()

cleaned_df.shape

"""Hasil data setelah di bersihkan adalah : 5332"""

df_movie.duplicated().sum()

"""Melihat jumlah niai unique yang ada di dalam dataset."""

print('Berikut adalah bentuk nilai unique dari setiap variabel :')
cleaned_df.nunique()

"""Membersihkan dan memastikan kolom `date_added` memiliki format date_time, sehingga siap digunakan untuk analisis waktu."""

#mengubah date update ke datetime format
# Hilangkan spasi ekstra
cleaned_df['date_added'] = cleaned_df['date_added'].str.strip()

# Konversi ke datetime dengan deteksi otomatis
cleaned_df['date_added'] = pd.to_datetime(cleaned_df['date_added'], format='mixed', errors='coerce')

# Cek nilai NaT setelah konversi
missing_dates = cleaned_df['date_added'].isna().sum()
print(f"Jumlah nilai yang tidak dapat dikonversi: {missing_dates}")

# Tangani nilai NaT
cleaned_df['date_added'].fillna('Unknown', inplace=True)

cleaned_df.info()

"""# 3 | Ekplolatory Data Analisis

Menampilkan grafik memberikan gambaran perbandingan jumlah antara kategori Movie dan TV Show dalam dataset. Visualisasi ini membantu memahami dominasi salah satu jenis konten dalam data.
"""

#type distribution
# Hitung distribusi
type_distribution = cleaned_df['type'].value_counts().reset_index()
type_distribution.columns = ['type', 'count']

# Membuat bar plot
sns.set(style="whitegrid")  # Mengatur gaya tampilan
plt.figure(figsize=(8, 5))
sns.barplot(x='type', y='count', data=type_distribution, palette='coolwarm' )

# Tambahkan elemen visual
plt.title('Distribusi Movie dan TV Show', fontsize=16, fontweight='bold')
plt.xlabel('Type', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

"""Sebagian besar data yang ada dalam dataset ini terdiri dari tayangan Movie, sementara TV Show terhitung sangat sedikit.

Menampilkan grafik histogram memberikan gambaran tentang distribusi jumlah rilis konten (baik Movie maupun TV Show) berdasarkan tahun rilisnya. Informasi ini membantu mengidentifikasi tren rilis konten dalam dataset.
"""

# Membuat figure
plt.figure(figsize=(12, 6))

# Plot histogram sebaran release_year untuk Movie dan TV Show
sns.histplot(data=cleaned_df, x='release_year', hue='type', kde=True, bins=30, palette='Set2')

# Tambahkan judul dan label
plt.title('Sebaran Tahun Rilis TV Show dan Movie', fontsize=16)
plt.xlabel('Tahun Rilis', fontsize=12)
plt.ylabel('Jumlah', fontsize=12)
plt.legend(title='Tipe', labels=['Movie', 'TV Show'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""Mayoritas tayangan dalam dataset dirilis pada dekade 2000-an hingga 2020-an, dengan dominasi Movie. Sebaran ini mencerminkan tren industri hiburan yang mengalami pertumbuhan besar dalam produksi konten selama dua dekade terakhir.

menampilkan grafik countplot yang memberikan distribusi jumlah konten untuk setiap kategori rating. Ini memudahkan analisis jenis rating apa yang paling umum atau jarang dalam dataset.
"""

plt.figure(figsize=(12, 6))

# Countplot untuk distribusi rating
sns.countplot(data=cleaned_df, y='rating',  order=cleaned_df['rating'].value_counts().index, palette='Set3')

# Tambahkan judul dan label
plt.title('Sebaran Rating Konten', fontsize=16)
plt.xlabel('Jumlah', fontsize=12)
plt.ylabel('Rating', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

"""Sebaran ini mencerminkan bahwa mayoritas konten dalam dataset ditargetkan untuk penonton dewasa dan remaja, sementara konten untuk anak-anak atau semua umur lebih jarang ditemukan. Hal ini mungkin mencerminkan tren industri hiburan yang cenderung memproduksi lebih banyak tayangan dengan tema dewasa.

menampilkan grafik barplot yang menunjukkan distribusi genre pada konten Netflix, membantu menganalisis genre mana yang paling banyak atau paling sedikit dalam dataset.
"""

# Pisahkan genre di kolom 'listed_in' berdasarkan koma dan buat daftar semua genre
genres = cleaned_df['listed_in'].str.split(',').explode().str.strip()

# Hitung frekuensi setiap genre
genre_counts = genres.value_counts()

# Visualisasikan sebaran genre menggunakan barplot
plt.figure(figsize=(14, 8))
sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='Set2')

# Tambahkan judul dan label
plt.title('Sebaran Genre pada Konten', fontsize=16)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Jumlah Konten', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""Dari sebaran dapat dilihat bahwa mayoritas konten adalah international movie diikuti drama dan komedi.

menampilkan grafik barplot yang menunjukkan distribusi konten berdasarkan negara. Ini memberikan gambaran tentang negara mana yang memiliki lebih banyak konten di platform Netflix.
"""

# Pisahkan negara di kolom 'country' berdasarkan koma dan buat daftar semua negara
countries = cleaned_df['country'].str.split(',').explode().str.strip()

# Hitung frekuensi setiap negara
country_counts = countries.value_counts()

# Visualisasikan sebaran negara menggunakan barplot
plt.figure(figsize=(20, 8))
sns.barplot(x=country_counts.index, y=country_counts.values, palette='Set3')

# Tambahkan judul dan label
plt.title('Sebaran Negara pada Konten', fontsize=16)
plt.xlabel('Negara', fontsize=12)
plt.ylabel('Jumlah Konten', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Pilih 10 negara teratas
top_10_countries = country_counts.head(10)

# Visualisasikan 10 negara teratas
plt.figure(figsize=(14, 8))
sns.barplot(x=top_10_countries.index, y=top_10_countries.values, palette='Set3')

# Tambahkan judul dan label
plt.title('10 Negara Teratas pada Konten', fontsize=16)
plt.xlabel('Negara', fontsize=12)
plt.ylabel('Jumlah Konten', fontsize=12)
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""Dari sebaran dapat dilihat bahwa movie paling banyak di produksi oleh United States kemudian di ikuti India dan United kingdom."""

# Tentukan 5 negara dengan konten terbanyak
top_countries = ['United States', 'India', 'United Kingdom', 'Canada', 'France']

# Filter dataset berdasarkan negara-negara yang telah ditentukan
filtered_df = cleaned_df[cleaned_df['country'].isin(top_countries)]

# Buat fungsi untuk mendapatkan top 10 aktor dan direktur dari setiap negara
def top_actors_directors(country_name):
    # Filter data berdasarkan negara
    country_data = filtered_df[filtered_df['country'].str.contains(country_name)]

    # Pisahkan aktor dan direktur, pastikan tidak ada missing values
    actors = country_data['cast'].dropna().str.split(',').explode().str.strip()
    directors = country_data['director'].dropna().str.split(',').explode().str.strip()

    # Hitung frekuensi aktor dan direktur
    top_actors = actors.value_counts().head(10)
    top_directors = directors.value_counts().head(10)

    return top_actors, top_directors

# Visualisasi hasil top 10 aktor dan direktur untuk setiap negara
for country in top_countries:
    top_actors, top_directors = top_actors_directors(country)

    # Plot untuk aktor
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top_actors.index, y=top_actors.values, palette='viridis')
    plt.title(f'Top 10 Aktor di {country}', fontsize=16)
    plt.xlabel('Aktor', fontsize=12)
    plt.ylabel('Jumlah Penampilan', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot untuk direktur
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top_directors.index, y=top_directors.values, palette='magma')
    plt.title(f'Top 10 Sutradara di {country}', fontsize=16)
    plt.xlabel('Sutradara', fontsize=12)
    plt.ylabel('Jumlah Penampilan', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

"""# 4 | Content base filtering

## 4.1| Data prep
"""

new_df = cleaned_df[['title','director','cast','listed_in','description']]
new_df.head()

"""Kode di bawah ini bertujuan untuk menghapus baris yang memiliki nilai kosong atau hanya berisi spasi pada kolom-kolom tertentu (`title`, `director`, `cast`, `listed_in`, `description`) dalam DataFrame `new_df`."""

blanks = []  # Mulai dengan daftar kosong untuk menyimpan indeks yang memiliki nilai kosong atau hanya spasi

col = ['title', 'director', 'cast', 'listed_in', 'description']  # Daftar kolom yang akan diperiksa

for i, col in new_df.iterrows():  # Iterasi baris-baris dalam DataFrame
    if type(col) == str:  # Memastikan nilai kolom adalah tipe string (untuk menghindari NaN)
        if col.isspace():  # Memeriksa apakah nilai tersebut hanya berisi spasi
            blanks.append(i)  # Jika ya, tambahkan indeks baris ke dalam daftar 'blanks'

new_df.drop(blanks, inplace=True)  # Hapus baris-baris yang memiliki indeks dalam 'blanks' dari DataFrame

"""Mengekstrak kata kunci penting (`key words`) dari deskripsi film atau acara TV di dataset menggunakan algoritma RAKE dan menyimpannya dalam kolom baru `Key_words`. Setelah itu, kolom description dihapus dari DataFrame karena sudah tidak diperlukan."""

import nltk

# Mengunduh model punkt
nltk.download('punkt', force=True)
nltk.download('punkt_tab')

"""mengekstraksi kata kunci (keywords) dari deskripsi konten yang ada dalam DataFrame new_df menggunakan library Rake dari rake_nltk."""

from rake_nltk import Rake

# Inisialisasi kolom Key_words sebagai daftar kosong
new_df['Key_words'] = [[] for _ in range(len(new_df))]

# Iterasi melalui DataFrame
for index, row in new_df.iterrows():
    description = row['description']

    # Menggunakan Rake untuk mengekstrak kata kunci
    r = Rake()  # By default, uses English stopwords from NLTK
    r.extract_keywords_from_text(description)

    # Mendapatkan kata kunci
    key_words_dict_scores = r.get_word_degrees()

    # Memperbarui kolom Key_words di DataFrame
    new_df.at[index, 'Key_words'] = list(key_words_dict_scores.keys())

# Menampilkan hasil
print(new_df[['title', 'Key_words']])

new_df.head()

# dropping the Plot column
new_df.drop(columns = ['description'], inplace = True)

new_df.head()

"""melakukan beberapa transformasi pada data dalam kolom cast, listed_in, dan director untuk membersihkan dan menyusun data sehingga lebih mudah diproses."""

# Menghapus koma di antara nama lengkap aktor dan hanya mengambil tiga nama pertama
new_df['cast'] = new_df['cast'].map(lambda x: x.split(',')[:3])

# Mengubah genre menjadi daftar kata dengan huruf kecil
new_df['listed_in'] = new_df['listed_in'].map(lambda x: x.lower().split(','))

# Memisahkan nama depan dan belakang sutradara
new_df['director'] = new_df['director'].map(lambda x: x.split(' '))

# Menggabungkan nama depan dan belakang setiap aktor dan sutradara menjadi satu kata,
# sehingga tidak ada kebingungan antara orang-orang dengan nama depan yang sama
for index, row in new_df.iterrows():
    row['cast'] = [x.lower().replace(' ','') for x in row['cast']]
    row['director'] = ''.join(row['director']).lower()

new_df.set_index('title', inplace = True)
new_df.head()

"""membuat representasi teks tunggal dari berbagai kolom dalam DataFrame dan menyimpannya dalam kolom baru bag_of_words."""

# Fungsi untuk menggabungkan semua kolom menjadi satu string dengan huruf kecil
def combine_columns(row):
    words = ''
    for col in new_df.columns:
        if col != 'director':
            # Jika kolom berupa daftar, gabungkan elemen dengan spasi
            if isinstance(row[col], list):
                words += ' '.join(row[col]).lower() + ' '  # Huruf kecil
            else:
                words += str(row[col]).lower() + ' '       # Huruf kecil
        else:
            # Gabungkan nama di kolom director jika berupa daftar
            if isinstance(row[col], list):
                words += ''.join(row[col]).lower() + ' '  # Gabungkan nama dengan huruf kecil
            else:
                words += str(row[col]).lower() + ' '      # Huruf kecil
    return words.strip()

# Terapkan fungsi untuk membuat kolom bag_of_words
new_df['bag_of_words'] = new_df.apply(combine_columns, axis=1)

# Hapus semua kolom kecuali bag_of_words
new_df = new_df[['bag_of_words']]

new_df.head()

"""## 4.2| Feature extraction and modeling

* Membuat dan menghasilkan matriks jumlah (count matrix) menggunakan CountVectorizer.
* Membuat sebuah Series untuk mencocokkan indeks dengan judul film, yang bisa digunakan nanti untuk referensi dalam proses pencocokan atau rekomendasi berbasis konten.
"""

# Membuat dan menghasilkan matriks jumlah (count matrix)
count = CountVectorizer()
count_matrix = count.fit_transform(new_df['bag_of_words'])

# Membuat Series untuk judul film sehingga judul-judul tersebut dihubungkan
# dengan daftar numerik berurutan. Daftar ini akan digunakan nanti untuk mencocokkan indeks.
indices = pd.Series(new_df.index)
indices[:5]

"""menghasilkan matriks kesamaan kosinus (cosine similarity matrix) antara semua film berdasarkan representasi teks yang telah dibuat menggunakan CountVectorizer."""

# Menghasilkan matriks kesamaan kosinus (cosine similarity matrix)
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Menampilkan matriks kesamaan kosinus
cosine_sim

# Fungsi yang menerima judul film sebagai input dan mengembalikan 10 rekomendasi film teratas
def recommendations(Title, cosine_sim=cosine_sim):

    recommended_movies = []  # Daftar untuk menyimpan rekomendasi film

    # Mendapatkan indeks film yang sesuai dengan judul
    idx = indices[indices == Title].index[0]

    # Membuat Series dengan skor kesamaan dalam urutan menurun
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # Mendapatkan indeks 10 film yang paling mirip
    top_10_indexes = list(score_series.iloc[1:11].index)

    # Mengisi daftar dengan judul dari 10 film yang paling sesuai
    for i in top_10_indexes:
        recommended_movies.append(list(new_df.index)[i])

    return recommended_movies

"""untuk mengevaluasi hasil rekomendasi dari model kita menggunakan precisi yang membagi berapa banyak rekomendasi yang relevan dengan jumlah total rekomendasi."""

recommendations('Adrift')

"""dari hasil diatas terdapat 9 movie yang sangat relevant dengan judul yang dicari sehingga precision dari hasil diatas adalah 90%"""

recommendations('Kung Fu Yoga')

"""dari hasil diatas terdapat 6 movie yang relevant dengan hasil rekomendasi sehingga precisi dari rekomendasi diatas adalah 60%

# 5 | Colaborative filtering

## 5.1 | Data prep

Membaca file data yang berisi informasi terkait pelanggan Netflix dan rating yang mereka berikan, kemudian mengonversi kolom Rating menjadi tipe data float.
"""

# Membaca file combined_data_1.txt dengan mengambil hanya dua kolom pertama (Customer ID dan Rating)
df1 = pd.read_csv(
    '/kaggle/input/netflix-prize-data/combined_data_1.txt',
    header=None,  # Tidak ada header pada file
    names=['Cust_Id', 'Rating'],  # Menamai kolom sebagai Cust_Id dan Rating
    usecols=[0, 1]  # Mengambil hanya kolom ke-0 dan ke-1
)

# Mengonversi kolom Rating menjadi tipe float
df1['Rating'] = df1['Rating'].astype(float)

df1.shape

df1.head()

df1.info()

df1.index = np.arange(0,len(df1))
df1.head()

p = df1.groupby('Rating')['Rating'].agg(['count'])

# get movie count
movie_count = df1.isnull().sum()[1]

# get customer count
cust_count = df1['Cust_Id'].nunique() - movie_count

# get rating count
rating_count = df1['Cust_Id'].count() - movie_count

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')

# Membuat DataFrame untuk menandai nilai NaN dalam kolom Rating
df_nan = pd.DataFrame(pd.isnull(df1.Rating))
df_nan = df_nan[df_nan['Rating'] == True]  # Memfilter hanya baris dengan Rating kosong
df_nan = df_nan.reset_index()  # Mengatur ulang indeks untuk proses lebih lanjut

# Inisialisasi variabel
movie_np = []  # Daftar untuk menyimpan ID film
movie_id = 1  # ID awal film

# Mengisi ID film untuk setiap segmen data
for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
    # Membuat array NumPy dengan panjang sesuai selisih indeks
    temp = np.full((1, i-j-1), movie_id)
    # Menambahkan array ke daftar movie_np
    movie_np = np.append(movie_np, temp)
    # Menambah ID film
    movie_id += 1

# Menangani data terakhir
last_record = np.full((1, len(df1) - df_nan.iloc[-1, 0] - 1), movie_id)
movie_np = np.append(movie_np, last_record)

# Menghapus baris dengan nilai NaN pada kolom Rating
df = df1[pd.notnull(df1['Rating'])]

# Menambahkan kolom Movie_Id ke DataFrame, dengan mengonversi movie_np menjadi tipe integer
df['Movie_Id'] = movie_np.astype(int)

# Mengonversi kolom Cust_Id ke tipe integer
df['Cust_Id'] = df['Cust_Id'].astype(int)

print(df.iloc[::5000000, :])

df.shape

# Statistik ulasan untuk setiap film
f = ['count', 'mean']  # Fungsi agregasi: hitung jumlah dan rata-rata
df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)

# Mengonversi indeks Movie_Id menjadi integer
df_movie_summary.index = df_movie_summary.index.map(int)

# Menentukan ambang batas ulasan film berdasarkan kuantil 80%
movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)

# Membuat daftar film yang memiliki jumlah ulasan di bawah ambang batas
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

# Mencetak ambang batas minimum ulasan untuk film
print('Movie minimum times of review: {}'.format(movie_benchmark))

# Statistik ulasan untuk setiap pelanggan
df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)

# Mengonversi indeks Cust_Id menjadi integer
df_cust_summary.index = df_cust_summary.index.map(int)

# Menentukan ambang batas ulasan pelanggan berdasarkan kuantil 80%
cust_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)

# Membuat daftar pelanggan yang memiliki jumlah ulasan di bawah ambang batas
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

# Mencetak ambang batas minimum ulasan untuk pelanggan
print('Customer minimum times of review: {}'.format(cust_benchmark))

# Menampilkan ukuran dataset sebelum pemotongan
print('Original Shape: {}'.format(df.shape))

# Menghapus baris yang mengandung Movie_Id dalam daftar drop_movie_list
df = df[~df['Movie_Id'].isin(drop_movie_list)]

# Menghapus baris yang mengandung Cust_Id dalam daftar drop_cust_list
df = df[~df['Cust_Id'].isin(drop_cust_list)]

# Menampilkan ukuran dataset setelah pemotongan
print('After Trim Shape: {}'.format(df.shape))

# Menampilkan contoh data dengan interval 5 juta baris
print('-Data Examples-')
print(df.iloc[::5000000, :])

df_p = pd.pivot_table(df,values='Rating',index='Cust_Id',columns='Movie_Id')

print(df_p.shape)

"""memuat file `movie_titles.csv`, yang berisi metadata tentang film, ke dalam DataFrame `df_title`"""

df_title = pd.read_csv(
    '/kaggle/input/netflix-prize-data/movie_titles.csv',
    encoding="ISO-8859-1",
    header=None,
    names=['Movie_Id', 'Year', 'Name'],
    on_bad_lines='skip'
)

print (df_title.head(10))

"""## 5.2 | Modeling"""

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()

# menggunakan 100.000 baris untuk waktu pelatihan lebih cepat
data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)

# Inisialisasi algoritma
algo = SVD()

# Evaluasi dengan cross-validation
results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Fungsi untuk memprediksi film yang mungkin disukai user
def recommend_movies(user_id, df_movies, n=10):
    # Mendapatkan daftar film yang belum ditonton user
    watched_movies = df[df['Cust_Id'] == user_id]['Movie_Id'].unique()
    movies_to_predict = df_movies[~df_movies['Movie_Id'].isin(watched_movies)]

    # Memprediksi skor untuk setiap film
    movies_to_predict['Estimate_Score'] = movies_to_predict['Movie_Id'].apply(
        lambda x: algo.predict(user_id, x).est
    )

    # Mengurutkan berdasarkan skor prediksi
    recommendations = movies_to_predict.sort_values('Estimate_Score', ascending=False)

    return recommendations[['Name', 'Year', 'Estimate_Score']].head(n)

recommended_movies = recommend_movies(user_id=78531, df_movies=df_title)
print(recommended_movies)