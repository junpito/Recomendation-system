import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate


def load_and_process_data(file_path):
    # Membaca data dan hanya mengambil dua kolom pertama
    df1 = pd.read_csv(file_path, header=None, names=['Cust_Id', 'Rating'], usecols=[0, 1])
    df1['Rating'] = df1['Rating'].astype(float)

    # Menandai data NaN pada kolom Rating
    df_nan = pd.DataFrame(pd.isnull(df1.Rating))
    df_nan = df_nan[df_nan['Rating'] == True].reset_index()

    # Menambahkan Movie_Id berdasarkan data NaN
    movie_np = []
    movie_id = 1
    for i, j in zip(df_nan['index'][1:], df_nan['index'][:-1]):
        movie_np = np.append(movie_np, np.full((1, i - j - 1), movie_id))
        movie_id += 1
    last_record = np.full((1, len(df1) - df_nan.iloc[-1, 0] - 1), movie_id)
    movie_np = np.append(movie_np, last_record)

    # Membersihkan data
    df = df1[pd.notnull(df1['Rating'])]
    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)

    return df


def filter_data(df):
    # Statistik untuk Movie_Id
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(['count', 'mean'])
    movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    # Statistik untuk Cust_Id
    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(['count', 'mean'])
    cust_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    # Filter data
    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]

    return df


def create_pivot_table(df):
    # Membuat pivot table
    return pd.pivot_table(df, values='Rating', index='Cust_Id', columns='Movie_Id')


def load_movie_titles(file_path):
    # Membaca metadata film
    return pd.read_csv(file_path, encoding="ISO-8859-1", header=None, names=['Movie_Id', 'Year', 'Name'], on_bad_lines='skip')


def train_model(df):
    # Melatih model menggunakan library Surprise
    reader = Reader()
    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:100000], reader)
    algo = SVD()
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return results


if __name__ == "__main__":
    # File paths
    data_path = '/kaggle/input/netflix-prize-data/combined_data_1.txt'
    titles_path = '/kaggle/input/netflix-prize-data/movie_titles.csv'

    # Load and process data
    df = load_and_process_data(data_path)

    # Filter data
    df = filter_data(df)

    # Pivot table (jika diperlukan)
    df_pivot = create_pivot_table(df)
    print(f"Pivot Table Shape: {df_pivot.shape}")

    # Load movie titles
    df_titles = load_movie_titles(titles_path)
    print(df_titles.head())

    # Train model
    results = train_model(df)
    print(results)