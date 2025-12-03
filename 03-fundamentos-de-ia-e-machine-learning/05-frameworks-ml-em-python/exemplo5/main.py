from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


# Criar a instância do LLM sem callback_manager (depreciado)
# ollama pull llama2
# ollama serve
# ollama rm llama2
llm = OllamaLLM(
    model="llama2",
    num_gpu=0
)

def gerar_insights_sobre_filmes(pergunta):
    prompt = f"Responda a seguinte pergunta sobre filmes: {pergunta}\n"
    prompt += "Por favor, forneça um resumo detalhado e quaisquer informações relevantes."

    # Usar callbacks via contexto (forma moderna)
    callbacks = [StreamingStdOutCallbackHandler()]
    insights = llm.invoke(prompt, config={"callbacks": callbacks})
    return insights

def responder_pergunta(pergunta):
    resposta = gerar_insights_sobre_filmes(pergunta)
    return resposta

def main():
    pergunta = input("Sobre qual filme você deseja saber informações? ")
    resposta = responder_pergunta(pergunta)
    print(f"Resposta: {resposta}")

if __name__ == "__main__":
    main()