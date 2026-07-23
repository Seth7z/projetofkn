# projetofk

# 📰 Detector de Fake News com Machine Learning e NLP

### 📝 Sobre o Projeto
Este é um projeto de Processamento de Linguagem Natural (NLP) focado na identificação automática de notícias falsas e reais. Desenvolvido como projeto final de curso técnico, o objetivo foi aplicar conceitos avançados de ciência de dados e inteligência artificial para solucionar o problema da desinformação.

### ⚙️ Como Funciona
O sistema processa textos jornalísticos, limpa as palavras irrelevantes (stop words) e aplica a técnica *TF-IDF* para transformar o texto em dados numéricos. O modelo compara dois algoritmos clássicos de classificação:
1. *PassiveAggressiveClassifier*
2. *LogisticRegression*

Ao final, o script utiliza *RandomizedSearchCV* para testar automaticamente dezenas de combinações de parâmetros (hiperparâmetros) e encontrar a versão mais precisa de cada modelo de IA.

### 🛠️ Tecnologias e Conceitos Aplicados
* *Linguagem*: Python
* *Manipulação de Dados*: Pandas e NumPy
* *Visualização*: Seaborn e Matplotlib (Gráficos de Matriz de Confusão)
* *Machine Learning*: Scikit-Learn (Pipelines, TF-IDF Vectorizer, Train-Test Split)
* *Métricas de Avaliação*: Acurácia, Matriz de Confusão e Classification Report (Precision, Recall, F1-Score)
