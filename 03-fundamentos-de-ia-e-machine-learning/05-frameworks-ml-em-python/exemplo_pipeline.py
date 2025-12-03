from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carregar um dataset exemplo
data = load_iris()
X, y = data.data, data.target

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

# Definir o pipeline
pipeline = Pipeline([
('scaler', StandardScaler()),
('pca', PCA(n_components=3)),
('classifier', RandomForestClassifier())
])

# Treinar o pipeline
pipeline.fit(X_train, y_train)

# Fazer previsões
predictions = pipeline.predict(X_test)

# Mostrar previsões
print(predictions)