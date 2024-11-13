import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

if not os.path.exists("imagens"):
    os.makedirs("imagens")


# Função para normalizar os dados com base no conjunto de treino
def normalizar_treino(X_train):
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    X_train_normalizado = (X_train - min_val) / (max_val - min_val)
    return X_train_normalizado, min_val, max_val


def normalizar_teste(X_test, min_val, max_val):
    X_test_normalizado = (X_test - min_val) / (max_val - min_val)
    return X_test_normalizado


# Função para calcular o RMSE
def calcular_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Carregar o conjunto de dados
data = pd.read_csv("boston.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir o conjunto de dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar apenas o conjunto de treino e aplicar a mesma escala ao conjunto de teste
X_train_normalizado, min_val, max_val = normalizar_treino(X_train)

X_test_normalizado = normalizar_teste(X_test, min_val, max_val)

# Listas para armazenar os RMSEs
train_rmse = []
test_rmse = []
train_rmse_ridge = []
test_rmse_ridge = []

# Treinamento dos modelos de regressão polinomial de ordem 1 a 11, sem e com regularização L2
for degree in range(1, 12):
    print(f"\n=== Treinando modelo de grau {degree} ===")

    # Transformação polinomial
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train_normalizado)
    X_test_poly = poly.transform(X_test_normalizado)

    # Regressão Polinomial (sem regularização)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Calcular RMSE para treino e teste
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    train_rmse_value = calcular_rmse(y_train, y_train_pred)
    test_rmse_value = calcular_rmse(y_test, y_test_pred)
    train_rmse.append(train_rmse_value)
    test_rmse.append(test_rmse_value)

    print(f"RMSE Treino (sem L2) para grau {degree}: {train_rmse_value}")
    print(f"RMSE Teste (sem L2) para grau {degree}: {test_rmse_value}")

    # Regressão Polinomial com Regularização L2 (Ridge Regression)
    ridge_model = Ridge(alpha=0.01)
    ridge_model.fit(X_train_poly, y_train)

    # Calcular RMSE para treino e teste com L2
    y_train_pred_ridge = ridge_model.predict(X_train_poly)
    y_test_pred_ridge = ridge_model.predict(X_test_poly)
    train_rmse_ridge_value = calcular_rmse(y_train, y_train_pred_ridge)
    test_rmse_ridge_value = calcular_rmse(y_test, y_test_pred_ridge)
    train_rmse_ridge.append(train_rmse_ridge_value)
    test_rmse_ridge.append(test_rmse_ridge_value)

    print(f"RMSE Treino (com L2) para grau {degree}: {train_rmse_ridge_value}")
    print(f"RMSE Teste (com L2) para grau {degree}: {test_rmse_ridge_value}")

# Plotar os resultados dos RMSEs para treino e teste (sem regularização)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), train_rmse, label="Treino RMSE (sem L2)", marker='o')
plt.plot(range(1, 12), test_rmse, label="Teste RMSE (sem L2)", marker='s')
plt.xlabel("Grau do Polinômio")
plt.ylabel("RMSE")
plt.title("RMSE para Treino e Teste por Grau de Polinômio (sem Regularização)")
plt.legend()
plt.savefig("imagens/rmse_sem_regularizacao.png")
plt.show()

# Plotar os resultados dos RMSEs para treino e teste (com regularização L2)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), train_rmse_ridge, label="Treino RMSE (com L2)", marker='o')
plt.plot(range(1, 12), test_rmse_ridge, label="Teste RMSE (com L2)", marker='s')
plt.xlabel("Grau do Polinômio")
plt.ylabel("RMSE")
plt.title("RMSE para Treino e Teste por Grau de Polinômio (com Regularização L2)")
plt.legend()
plt.savefig("imagens/rmse_com_regularizacao_l2.png")
plt.show()

