from sklearn import cross_validation as cv
import pandas as pd
import numpy as np

# загружаем датасет
df = pd.read_csv('../ml-latest-small/ratings.csv')

# выводим количество пользователей и фильмов
n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]

# чтобы можно было удобно работать дальше, необходимо отмасштабировать
# значения в колонке movieId (новые значения будут в диапазоне от 1 до
# количества фильмов)
input_list = df['movieId'].unique()

def scale_movie_id(input_id):
    return np.where(input_list == input_id)[0][0] + 1

df['movieId'] = df['movieId'].apply(scale_movie_id)

# делим данные на тренировочный и тестовый наборы
train_data, test_data = cv.train_test_split(df, test_size=0.20)

# создаём две user-item матрицы – для обучения и для теста
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
