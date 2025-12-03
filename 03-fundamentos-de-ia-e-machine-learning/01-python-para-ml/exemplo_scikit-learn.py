# Treinar um modelo de regressão linear.
# Previsão de preços imobiliários com base em características como tamanho e localização.

from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression().fit(X, y)
print(model.coef_)