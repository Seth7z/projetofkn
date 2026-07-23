# -*- coding: utf-8 -*-
"""Projeto de fake news"""

# Importando as bibliotecas
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Lendo os dados
df = pd.read_csv('/content/news.csv')

# Exibindo os dados
print("Shape do dataset", df.shape)
print(df.head())

# Separando features e labels
X = df['text']
y = df['label']

#Dividindo os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Criando pipeline para o modelo PassiveAggressiveCLassifier
pipeline_pac = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('classifier', PassiveAggressiveClassifier(max_iter=50))
])

# Treinando o modelo
pipeline_pac.fit(x_train, y_train)

# Fazendo previsões
y_pred_pac = pipeline_pac.predict(x_test)

# Avaliação do modelo PassiveAggressiveClassifier
print("\n### PassiveAggressiveClassifier ###")
print(f'Acurácia: {round(accuracy_score(y_test, y_pred_pac) * 100, 2)}%')
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_pac))
print("Relatório de classificação:\n", classification_report(y_test, y_pred_pac))

# Criando pipeline para o modelo LogisticRegression
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Treinando o modelo
pipeline_lr.fit(x_train, y_train)

# Fazendo previsôes
y_pred_lr = pipeline_lr.predict(x_test)

# Avaliaçãon do modelo LogisticRegression
print("\n### LogisticRegression ###")
print(f'Acurácia: {round(accuracy_score(y_test, y_pred_lr) * 100, 2)}%')
print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred_lr))
print("Relatório de classificação:\n", classification_report(y_test, y_pred_lr))

# Função para exibir matriz de confusão
def plot_confusion_matrix(y_true, y_pred, title):
  cm = confusion_matrix(y_true, y_pred, labels=['FAKE', 'REAL'])
  plt.figure(figsize=(5, 4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
  plt.xlabel('Previsão')
  plt.ylabel('Verdadeiro')
  plt.title(title)
  plt.show()

# Visualizando as matrizes de confusão
plot_confusion_matrix(y_test, y_pred_pac, "Matriz de Confusão - PassiveAggressiveClassifier")
plot_confusion_matrix(y_test, y_pred_lr, "Matriz de Confusão - LogisticRegression")

# Exibir algumas previsões
def mostrar_previsao(X_test, y_test, y_pred, num=5):
  df_results = pd.DataFrame({'Texto': X_test[:num], 'Real': y_test[:num], 'Previsto': y_pred[:num]})
  print("\n### Exemplos de previsões ###")
  print(df_results)

mostrar_previsao(X_test.values, y_test.values, y_pred_lr)

param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100], # Regularização
    'classifier__max_iter': [500, 1000, 1500], # Número de iterações
    'classifier__solver': ['lbfgs', 'liblinear'] # Algoritmo de otimização
}

param_grid_pac = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Regularização
    'classifier__max_iter': [50, 100, 200],  # Número máximo de iterações
    'classifier__tol': [1e-3, 1e-4, 1e-5]  # Tolerância para convergência
}

# Aplicando RandomizedSearchCV no Logistic Regression
random_search_lr = RandomizedSearchCV(pipeline_lr, param_grid_lr, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_lr.fit(x_train, y_train)

# Melhor modelo e hiperparâmetros encontrados
print("Melhores hiperparâmetros para Logistic Regression:", random_search_lr.best_params_)
print("Melhor acurácia no treino (LR)", random_search_lr.best_score_)

# Aplicando RandomizedSearchCV no PassiveAggressiveClassifier
random_search_pac = RandomizedSearchCV(pipeline_pac, param_grid_pac, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search_pac.fit(X_train, y_train)

# Melhor modelo e hiperparâmetros encontrados
print("Melhores hiperparâmetros para Passive Aggressive Classifier:", random_search_pac.best_params_)
print("Melhor acurácia no treino (PAC):", random_search_pac.best_score_)

# Fazendo previsões com os melhores modelos encontrados
y_pred_lr_best = random_search_lr.best_estimator_.predict(X_test)
y_pred_pac_best = random_search_pac.best_estimator_.predict(X_test)

# Avaliação
print("Acurácia do melhor modelo Logistic Regression:", accuracy_score(y_test, y_pred_lr_best))
print("Acurácia do melhor modelo Passive Aggressive Classifier:", accuracy_score(y_test, y_pred_pac_best))
