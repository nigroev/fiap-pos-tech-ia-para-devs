from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


composto1 = [1, 1, 1]
composto2 = [0, 0, 0]
composto3 = [1, 0, 1]
composto4 = [0, 1, 0]
composto5 = [1, 1, 0]
composto6 = [0, 0, 1]

dados_treino = [composto1, composto2, composto3, composto4, composto5, composto6]
rotulos_treino = ['S', 'N', 'S', 'N', 'S', 'S']

modelo = LinearSVC()
modelo.fit(dados_treino, rotulos_treino)

teste1 = [1, 0, 0]
teste2 = [0, 1, 1]
teste3 = [1, 0, 1]

dados_teste = [teste1, teste2, teste3]
rotulos_teste = ['S', 'N', 'S']

previsoes = modelo.predict(dados_teste)

taxa_acerto = accuracy_score(rotulos_teste, previsoes)
print("Taxa de acerto: %.2f%%" % (taxa_acerto * 100))