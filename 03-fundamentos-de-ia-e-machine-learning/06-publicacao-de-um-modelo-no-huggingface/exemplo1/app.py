from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Carregar o tokenizador e o modelo pré-treinado
tokenizer = AutoTokenizer.from_pretrained("tadrianonet/distilbert-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("tadrianonet/distilbert-text-classification")

# Função de previsão
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label, probabilities

# Testar vários textos positivos e neutros para verificar a resposta do modelo
texts_to_test = [
    "Estou extremamente feliz com os resultados que conseguimos! Foi um trabalho incrível de toda a equipe!",
    "O filme de ontem foi incrível e estou muito empolgado com o novo projeto na empresa!",
    "A comida estava deliciosa e o atendimento foi excelente. Vou recomendar para todos os meus amigos!",
    "Hoje foi um dia maravilhoso cheio de boas notícias e realizações!",
    "A reunião foi muito produtiva e todos estão animados para os próximos passos!."
]

# Definição das classes
classes = ["Negativo/Neutro", "Positivo"]

# Realizar previsões para cada texto
for text in texts_to_test:
    predicted_label, probabilities = predict(text)
    print(f"Texto: {text}")
    print(f"Rótulo Previsto: {predicted_label} ({classes[predicted_label]})")
    print(f"Probabilidades: {probabilities}\n")
    