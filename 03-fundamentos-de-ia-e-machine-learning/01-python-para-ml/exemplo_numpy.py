# Exemplo: Calcular a média de uma matriz de dados.
# Caso de uso: Processamento de imagens onde essas figuras são representadas como matrizes de pixels. 

import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = np.mean(data)

print(mean)