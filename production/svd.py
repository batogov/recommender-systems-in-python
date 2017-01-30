import scipy.sparse as sp
from scipy.sparse.linalg import svds
from rmse import rmse
import numpy as np

from work_with_data import train_data_matrix, test_data_matrix, n_users, n_items

# делаем SVD
u, s, vt = svds(train_data_matrix, k=10)
s_diag_matrix = np.diag(s)

# предсказываем
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

# выводим метрику
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))
