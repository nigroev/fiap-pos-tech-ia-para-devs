# Exemplo: criar um gráfico de distribuição.
# Caso de Uso: Análise de distribuição de dados demográficos em pesquisas sociais.

import seaborn as sns
import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
sns.histplot(data)
plt.show()