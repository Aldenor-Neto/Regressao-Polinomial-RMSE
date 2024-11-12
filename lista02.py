import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

# Verifique se a pasta "imagens" existe, caso contrário, crie-a
if not os.path.exists("imagens"):
    os.makedirs("imagens")

# Carregar o conjunto de dados
data = pd.read_csv("boston.csv")
X = data.iloc[:, :-1].values  # Atributos (13 primeiras colunas)
y = data.iloc[:, -1].values   # Alvo (preço das casas, última coluna)

# Dividir o conjunto de dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Listas para armazenar os RMSEs
train_rmse = []
test_rmse = []
train_rmse_ridge = []
test_rmse_ridge = []

# Treinamento dos modelos de regressão polinomial de ordem 1 a 11, sem e com regularização L2
for degree in range(1, 12):
    # Transformação polinomial
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Regressão Polinomial (sem regularização)
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Calcular RMSE para treino e teste
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    train_rmse.append(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    test_rmse.append(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    
    # Regressão Polinomial com Regularização L2 (Ridge Regression)
    ridge_model = Ridge(alpha=0.01)
    ridge_model.fit(X_train_poly, y_train)
    
    # Calcular RMSE para treino e teste com L2
    y_train_pred_ridge = ridge_model.predict(X_train_poly)
    y_test_pred_ridge = ridge_model.predict(X_test_poly)
    train_rmse_ridge.append(np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)))
    test_rmse_ridge.append(np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)))

# Plotar os resultados dos RMSEs para treino e teste (sem regularização)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), train_rmse, label="Treino RMSE (sem L2)", marker='o')
plt.plot(range(1, 12), test_rmse, label="Teste RMSE (sem L2)", marker='s')
plt.xlabel("Grau do Polinômio")
plt.ylabel("RMSE")
plt.title("RMSE para Treino e Teste por Grau de Polinômio (sem Regularização)")
plt.legend()
plt.savefig("imagens/rmse_sem_regularizacao.png")  # Salva a imagem
plt.show()

# Plotar os resultados dos RMSEs para treino e teste (com regularização L2)
plt.figure(figsize=(12, 6))
plt.plot(range(1, 12), train_rmse_ridge, label="Treino RMSE (com L2)", marker='o')
plt.plot(range(1, 12), test_rmse_ridge, label="Teste RMSE (com L2)", marker='s')
plt.xlabel("Grau do Polinômio")
plt.ylabel("RMSE")
plt.title("RMSE para Treino e Teste por Grau de Polinômio (com Regularização L2)")
plt.legend()
plt.savefig("imagens/rmse_com_regularizacao_l2.png")  # Salva a imagem
plt.show()
