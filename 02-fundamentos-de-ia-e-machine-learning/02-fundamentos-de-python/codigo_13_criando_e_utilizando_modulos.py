# Criando e Utilizando Módulos:

# em meu_modulo.py
def soma(a, b):
    return a + b

# em outro arquivo
from meu_modulo import soma
print(soma(3, 4))  # 7