# 🏠 Housing Price Prediction

Projeto de Machine Learning focado na predição de preços de imóveis utilizando técnicas de regressão e pipelines de dados.

## 📌 Objetivo

O objetivo deste projeto é construir um modelo capaz de prever o preço de imóveis com base em diferentes características (features), explorando desde a análise dos dados até a avaliação do modelo final.

Este projeto foi desenvolvido com foco em prática de conceitos de Data Science e Machine Learning aplicados a problemas reais.

Pequeno spoiler do resultado:

<p align="center">
  <img src="https://raw.githubusercontent.com/raphamonn/housing-regression/main/imgs/result.png" alt="Gráfico de Resultados" width="600">
</p>

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
- RMSE *(Root Mean Squared Error)*: **47517.1043** | **42841.7729** (Removendo Outliers)
- R² *(Coeficiente de determinação)* : **0.8267** | **0.8006** (Removendo Outliers)
- MSE *(Mean Square Error)* : **2257875197.1973** | **1835417509.4771** (Removendo Outliers)
- MAE *(Mean Absolute Error)*: **31469.6485** | **28816.9436** (Removendo Outliers)
---

## 📁 Estrutura do projetoE

```text
housing-regression/
├── datasets/
    └──  housing/            # local de armazenamento de dados crus     
├── imgs/                   # gráficos e imagens geradas
├── models/                 # Local onde os modelos são salvos como arquivos .pkl 
├── src/                    # Código-fonte auxiliar e scripts Python
├── notebooks/              # Notebooks Jupyter com as análises e experimentos                    
├── README.md               # Documentação do projeto
└── requirements.txt        # Lista de dependências e bibliotecas
```
---
## 🧠 Aprendizados

O projeto foi desenvolvido com objetivo de aprendizado, mas estruturado como um caso de negócio real.
Supondo uma empresa fictícia chamada California Houses, que atua no mercado imobiliário, existe a necessidade de melhorar o processo de precificação de imóveis.
Atualmente, os preços são definidos com base na experiência dos corretores e comparações manuais, o que gera inconsistências, erros de avaliação e perda de oportunidades de negócio.
Então a ideia foi criar um modelo que fosse o mais fidedigno ao que já temos, mas entender como isso pode auxiliar nas ofertas.


### Então, agora muito mais aprofundado, quero responder algumas perguntas:
- Qual é o risco do modelo estar performando mal;
```text
Se tratando da precificação de imóveis, o problema é bem definido, 
```

- Quais imóveis estão subvalorizados (são oportunidades de compra);
- Quais características mais impactaram a precificação;
- Precificação assistida: dá pra usar o modelo como um "miniapp" com inputação de dados;
- Em quais faixa o modelo de preço performa melhor ? Imóveis mais caros mais baratos ou medianos? 

