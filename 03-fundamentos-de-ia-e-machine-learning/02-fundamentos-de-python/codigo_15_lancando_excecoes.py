# Lançando Exceções:

def dividir(a, b):
    if b == 0:
        raise ValueError("O divisor não pode ser zero.")
    return a / b

try:
    print(dividir(10, 0))
except ValueError as e:
    print(e)