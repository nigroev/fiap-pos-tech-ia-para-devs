# Criar e treinar uma rede neural simples
# Reconhecimento de imagem para classificação de objetos em fotos.

import tensorflow as tf
import numpy as np

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 3 características
# y: vetor de saída com 100 valores
X = np.random.random((100, 3))
y = np.random.random((100, 1))

# Definindo o modelo usando Input
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(10, activation='relu')(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilando o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinando o modelo
model.fit(X, y, epochs=5)