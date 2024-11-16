# Relatório de Regressão Polinomial

## Instituto Federal do Ceará, Campus Maracanaú  

**Disciplina:** Reconhecimento de Padrões  
**Professor:** Hericson Araújo  
**Aluno:** Francisco Aldenor Silva Neto  
**Matrícula:** 20221045050117  



## Introdução

Este relatório descreve a aplicação de modelos de regressão polinomial para prever os preços das casas em Boston utilizando o conjunto de dados **boston.csv**. O objetivo é treinar modelos de regressão polinomial variando a ordem dos polinômios de 1 a 11 e avaliar o desempenho desses modelos com e sem regularização L2 (Ridge Regression).

A regressão polinomial é uma extensão da regressão linear que busca modelar relações não lineares entre as variáveis independentes e dependentes, ajustando um polinômio de grau \(d\). Para evitar o sobreajuste (overfitting), utilizamos a regularização L2, que adiciona uma penalidade aos coeficientes do modelo.

### Fórmula da Regressão Polinomial
g
A regressão polinomial busca minimizar o erro quadrático entre os valores previstos e os reais. A função objetivo é definida como:

```
ŷ = β₀ + β₁x + β₂x² + ... + βdxᵈ
```

onde:  
- `x` são as variáveis independentes,  
- `β₀, β₁, ..., βd` são os coeficientes do modelo,  
- `d` é o grau do polinômio.  

A função de custo (erro quadrático médio) é dada por:  

```
J(β) = (1/n) * Σᵢ (yᵢ - ŷᵢ)²
```

### Regularização L2

Para evitar que o modelo ajuste demais os dados de treino (overfitting), a regularização L2 adiciona um termo de penalização baseado na soma dos quadrados dos coeficientes:

```
J_Ridge(β) = (1/n) * Σᵢ (yᵢ - ŷᵢ)² + λ * Σⱼ βⱼ²
```

onde:  
- `λ` é o parâmetro de regularização que controla a intensidade da penalização,  
- `Σⱼ βⱼ²` é o termo de regularização que penaliza coeficientes grandes.  

Com `λ = 0`, a regularização não é aplicada, resultando na regressão polinomial padrão. Para `λ > 0`, a regularização L2 reduz a magnitude dos coeficientes, limitando a complexidade do modelo.


## Conjunto de Dados

O conjunto de dados **boston.csv** contém 14 colunas, sendo 13 atributos (independentes) e 1 variável de saída (dependente), que corresponde ao preço das casas em Boston na década de 1970.

---

## Metodologia

### 1. Divisão dos Dados

Os dados foram divididos aleatoriamente em dois conjuntos:
- **Treinamento (80%)**
- **Teste (20%)**

### 2. Normalização dos Dados

Os dados de entrada foram normalizados utilizando a faixa entre o menor e o maior valor dos dados de treinamento.

### 3. Modelos de Regressão Polinomial

Foram treinados modelos de regressão polinomial de grau 1 a 11, tanto com quanto sem regularização L2. O processo de treinamento foi realizado utilizando a biblioteca **scikit-learn** para manipulação de dados e modelos.

### 4. Regularização L2

A regularização L2 foi aplicada utilizando o modelo de **Ridge Regression**, com um valor de **lambda** igual a 0,01.

### 5. Avaliação dos Modelos

A performance dos modelos foi avaliada utilizando o **RMSE** (Raiz Quadrada do Erro Quadrático Médio) para as previsões tanto no conjunto de treino quanto no conjunto de teste.

## Resultados

### RMSE sem Regularização L2

![RMSE sem Regularização L2](imagens/Lambda%20=%200,01/rmse_sem_regularizacao.png)

| Grau do Polinômio | RMSE Treinamento | RMSE Teste |
|-------------------|------------------|------------|
| 1                 | 4.7689           | 4.4022     |
| 2                 | 2.4531           | 3.0390     |
| 3                 | 4.7734e-11       | 169.5561   |
| 4                 | 1.8117e-12       | 32.0649    |
| 5                 | 6.0808e-13       | 20.4379    |
| 6                 | 5.2788e-13       | 15.2600    |
| 7                 | 4.7599e-13       | 12.9143    |
| 8                 | 1.0704e-12       | 11.7193    |
| 9                 | 1.9051e-12       | 10.9604    |
| 10                | 5.1371e-12       | 10.4131    |
| 11                | 1.2502e-11       | 10.0519    |

Observa-se que o overfitting começa a se tornar evidente a partir do grau 3. A partir desse ponto, o RMSE no conjunto de treino sem regularização se aproxima de zero, enquanto o RMSE no conjunto de teste aumenta drasticamente, indicando que o modelo está ajustando ruídos ao invés de padrões reais. Isso se confirma nos graus seguintes, onde o RMSE de treino permanece próximo de zero e o RMSE de teste se estabiliza em valores elevados, caracterizando um sobreajuste severo.

### RMSE com Regularização L2 (lambda = 0,01)

![RMSE com Regularização L2](imagens/Lambda%20=%200,01/rmse_com_regularizacao_l2.png)

| Grau do Polinômio | RMSE Treinamento | RMSE Teste |
|-------------------|------------------|------------|
| 1                 | 4.7689           | 4.4011     |
| 2                 | 2.5822           | 3.0646     |
| 3                 | 2.0187           | 2.4092     |
| 4                 | 1.6258           | 3.1436     |
| 5                 | 1.3038           | 4.2137     |
| 6                 | 1.1002           | 4.5990     |
| 7                 | 0.9572           | 4.6498     |
| 8                 | 0.8457           | 4.6443     |
| 9                 | 0.7557           | 4.6788     |
| 10                | 0.6816           | 4.7489     |
| 11                | 0.6201           | 4.8755     |

Com a regularização L2, observa-se que o RMSE no conjunto de treino não atinge valores próximos de zero para graus maiores, mantendo-se mais próximo ao RMSE do conjunto de teste. A regularização reduz o efeito de sobreajuste observado na versão sem regularização, porém ainda há indícios de overfitting a partir de polinômios de grau 5, uma vez que o RMSE de teste se estabiliza e não melhora significativamente, enquanto o RMSE de treino continua a diminuir levemente. Isso indica que, apesar da regularização, os modelos de graus mais altos ainda capturam padrões específicos do conjunto de treino que não generalizam bem para o conjunto de teste.

### **Fins de Comparação: Análise dos Efeitos de Diferentes Valores de Regularização**

Para avaliar os impactos da regularização L2 no comportamento dos modelos e sua capacidade de resolver o problema de overfitting, foram realizados testes com os valores de \( \lambda = 0.1 \) e \( \lambda = 10 \). O objetivo foi observar se ajustes na penalidade poderiam reduzir a discrepância entre os erros de treino e teste, garantindo uma melhor generalização.

####  RMSE com Regularização L2 (lambda = 0,1)

O gráfico gerado a partir desse teste está apresentado na **Figura abaixo**, e a tabela seguinte lista os RMSEs obtidos.

![RMSE com regularização L2 - lambda = 0.1 ](imagens/Lambda%20=%201/rmse_com_regularizacao_l2.png)

**Tabela 4 - RMSE para diferentes graus do modelo com \( \lambda = 0.1 \):**
| Grau do Modelo | RMSE Treino | RMSE Teste |
|----------------|-------------|------------|
| 1              | 5.5030      | 4.2896     |
| 2              | 4.5359      | 3.8585     |
| 3              | 3.8053      | 3.3365     |
| 4              | 3.3261      | 3.1385     |
| 5              | 3.0212      | 3.0363     |
| 6              | 2.8434      | 2.9713     |
| 7              | 2.7220      | 2.9368     |
| 8              | 2.6137      | 2.9222     |
| 9              | 2.5045      | 2.9258     |
| 10             | 2.3953      | 2.9495     |
| 11             | 2.2920      | 2.9910     |

---

#### **Resultados para \( \lambda = 10 \):**
Na sequência, foi realizado o mesmo experimento com \( \lambda = 10 \), cujos resultados estão ilustrados na **Figura abaixo** e descritos na tabela seguinte.

**Figura - Gráfico de RMSE com regularização L2 (\( \lambda = 10 \)):**
![RMSE com regularização L2 - lambda = 10 ](imagens/Lambda%20=%2010/rmse_com_regularizacao_l2.png)

**Tabela 5 - RMSE para diferentes graus do modelo com \( \lambda = 10 \):**
| Grau do Modelo | RMSE Treino | RMSE Teste |
|----------------|-------------|------------|
| 1              | 5.5030      | 4.2896     |
| 2              | 4.5359      | 3.8585     |
| 3              | 3.8053      | 3.3365     |
| 4              | 3.3261      | 3.1385     |
| 5              | 3.0212      | 3.0363     |
| 6              | 2.8434      | 2.9713     |
| 7              | 2.7220      | 2.9368     |
| 8              | 2.6137      | 2.9222     |
| 9              | 2.5045      | 2.9258     |
| 10             | 2.3953      | 2.9495     |
| 11             | 2.2920      | 2.9910     |

#### **Discussão dos Resultados:**
Ao aumentar \( \lambda \) de 0.1 para 10, observou-se uma diminuição do efeito de overfitting, especialmente em modelos de grau elevado. A regularização mais intensa suprime coeficientes polinomiais excessivos, melhorando a estabilidade dos resultados no conjunto de teste, mas com aumento no erro no conjunto de treino. Essa troca reflete o compromisso entre bias e variância, típico em problemas de regressão com regularização.



## Conclusão

Os resultados demonstram que a regularização L2 ajuda a controlar o sobreajuste, especialmente para polinômios de grau mais elevado, ao limitar o ajuste excessivo aos dados de treino. No entanto, o efeito de overfitting ainda é observado para graus maiores que 4, indicando que, apesar da regularização, esses modelos ainda capturam ruídos específicos do conjunto de treino.

Sem regularização, o modelo apresenta um ajuste quase perfeito no treino para graus elevados, mas com um aumento drástico do erro no conjunto de teste, evidenciando um sobreajuste severo. A regularização L2 reduz o impacto desse comportamento, estabilizando os erros de teste, mas não eliminando completamente o efeito.

### Observações

- Para graus mais baixos (1 e 2), o desempenho dos modelos com e sem regularização é semelhante, pois a complexidade do modelo é gerenciável.
- A regularização L2 limita o ajuste excessivo em graus mais altos, melhorando a estabilidade dos modelos. No entanto, o efeito de overfitting ainda persiste para modelos mais complexos (graus acima de 4).

---

## Repositório no GitHub

O código fonte deste trabalho está disponível no seguinte repositório:  
[Regressão Polinomial - RMSE](https://github.com/Aldenor-Neto/Regressao-Polinomial-RMSE)
