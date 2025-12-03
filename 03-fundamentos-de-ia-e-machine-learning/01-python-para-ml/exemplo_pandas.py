# Exemplo: carregar um arquivo CSV e calcular a média de uma coluna.
# Caso de uso: Análise de dados financeiros para calcular médias e tendências de preços.

import pandas as pd

data = pd.read_csv('data.csv')
mean_value = data['column_name'].mean()

print(mean_value)