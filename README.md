# 🏠 Housing Price Prediction

Projeto de Machine Learning focado na predição de preços de imóveis utilizando técnicas de regressão e pipelines de dados.

## 📌 Objetivo

O objetivo deste projeto é construir um modelo capaz de prever o preço de imóveis com base em diferentes características (features), explorando desde a análise dos dados até a avaliação do modelo final.

Este projeto foi desenvolvido com foco em prática de conceitos de Data Science e Machine Learning aplicados a problemas reais.

Pequeno spoiler do resultado:
![Gráfico exibindo um ótimo resultado, onde o R2 está bem próximo de 1](img/result.png)

---

## 🧠 Abordagem

O projeto segue um pipeline completo de Machine Learning:

1. **Análise Exploratória (EDA)**
   - Entendimento das variáveis
   - Identificação de padrões e correlações

2. **Pré-processamento**
   - Tratamento de valores nulos
   - Feature engineering
   - Encoding de variáveis categóricas
   - Normalização / padronização

3. **Modelagem**
   - Treinamento de modelos de regressão
   - Uso de pipelines do Scikit-Learn

4. **Validação**
   - Separação treino/teste
   - Cross-validation
   - Avaliação com métricas de regressão
   - Uso de RandomSearch e GridSearch para determinar a melhor combinação de parâmetros
   - Melhor modelo *RandomSearch* obteve **RMSE** de aproximadamente **49.313** (cross-validation)

---

## ⚙️ Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## 📊 Métricas de avaliação

O modelo foi avaliado utilizando:
- RMSE *(Root Mean Squared Error)*: **47517.1043**
- R² *(Coeficiente de determinação)* : **0.8267**
- MSE *(Mean Square Error)* : **2257875197.1973**
- MAE *(Mean Absolute Error)*: **31469.6485**
---

## 📁 Estrutura do projeto
