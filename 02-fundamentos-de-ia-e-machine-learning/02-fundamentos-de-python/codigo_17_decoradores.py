# Exemplo de Decoradores:

def decorador_saudacao(func):
    def wrapper(*args, **kwargs):
        print("Saudação!")
        return func(*args, **kwargs)
    return wrapper

@decorador_saudacao
def ola(nome):
    print(f"Olá, {nome}!")

ola("Mundo")  # Saudação! Olá, Mundo!