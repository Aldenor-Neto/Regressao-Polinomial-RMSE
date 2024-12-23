import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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
    erro_quadratico = np.mean((y_true - y_pred)**2)
    return np.sqrt(erro_quadratico)

# Função para calcular os coeficientes beta (sem regularização, com proteção contra matrizes singulares)
def calcular_coeficientes(X, y, alpha=1e-8):
    # Adiciona uma coluna de 1s para o termo de bias (intercepto)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    # Protege contra a singularidade da matriz adicionando um pequeno valor à diagonal
    XtX = X_b.T.dot(X_b)
    XtX += np.eye(XtX.shape[0]) * alpha
    # Calcula os coeficientes beta usando a fórmula dos mínimos quadrados
    beta = np.linalg.inv(XtX).dot(X_b.T).dot(y)
    return beta

# Função para criar as características polinomiais
def gerar_caracteristicas_polinomiais(X, grau):
    X_poly = X
    for i in range(2, grau+1):
        X_poly = np.c_[X_poly, X**i]
    return X_poly

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

# Treinamento dos modelos de regressão polinomial
for degree in range(1, 12):
    print(f"\n=== Treinando modelo de grau {degree} ===")

    # Transformação polinomial
    X_train_poly = gerar_caracteristicas_polinomiais(X_train_normalizado, degree)
    X_test_poly = gerar_caracteristicas_polinomiais(X_test_normalizado, degree)

    # Regressão Polinomial (sem regularização)
    beta = calcular_coeficientes(X_train_poly, y_train)

    # Previsão para treino e teste
    y_train_pred = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly].dot(beta)
    y_test_pred = np.c_[np.ones((X_test_poly.shape[0], 1)), X_test_poly].dot(beta)

    # Calcular RMSE para treino e teste
    train_rmse_value = calcular_rmse(y_train, y_train_pred)
    test_rmse_value = calcular_rmse(y_test, y_test_pred)
    train_rmse.append(train_rmse_value)
    test_rmse.append(test_rmse_value)

    print(f"RMSE Treino (sem L2) para grau {degree}: {train_rmse_value}")
    print(f"RMSE Teste (sem L2) para grau {degree}: {test_rmse_value}")

    # Regressão Polinomial com Regularização L2
    # Função para calcular beta com regularização L2
    def calcular_coeficientes_ridge(X, y, alpha=0.1):
        # Adiciona uma coluna de 1s para o termo de bias (intercepto)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        XtX = X_b.T.dot(X_b)
        XtX += np.eye(XtX.shape[0]) * alpha
        beta = np.linalg.inv(XtX).dot(X_b.T).dot(y)
        return beta

    # Calcular coeficientes com regularização L2
    beta_ridge = calcular_coeficientes_ridge(X_train_poly, y_train, alpha=0.01)

    # Previsão para treino e teste com regularização L2
    y_train_pred_ridge = np.c_[np.ones((X_train_poly.shape[0], 1)), X_train_poly].dot(beta_ridge)
    y_test_pred_ridge = np.c_[np.ones((X_test_poly.shape[0], 1)), X_test_poly].dot(beta_ridge)

    # Calcular RMSE para treino e teste com L2
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
