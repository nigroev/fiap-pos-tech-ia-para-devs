# Exemplo: criar um modelo de classificação com Keras.
# Caso de uso: detecção de fraude em transações financeiras com base em padrões de dados.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 8 características
# y: vetor de saída com 100 valores binários (0 ou 1)
X = np.random.random((100, 8))
y = np.random.randint(2, size=(100, 1))

# Definindo o modelo
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compilando o modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinando o modelo
model.fit(X, y, epochs=150, batch_size=10)