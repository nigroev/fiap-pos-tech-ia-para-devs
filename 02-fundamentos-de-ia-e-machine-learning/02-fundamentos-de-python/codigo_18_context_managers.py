# Exemplo de Context Managers:

class GerenciadorDeContexto:
    def __enter__(self):
        print("Entrando no contexto.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Saindo do contexto.")

with GerenciadorDeContexto():
    print("Dentro do bloco with.")