'''
data_exploration.py

Primeiro contato com o Dataset. O objetivo é conhecer com q estou trabalhando. Descobrir o que precisa ser feito de tratamento de dados e as principais features.

Resumo:
    Temos 284315 uso legal que representa 99.83% de nosso Dataset.
    Temos 492 uso fraudulento que representa 0.17% de nosso Dataset.
    E o valor médio do uso do cartão de crédito é de 88,34.
    Não temos nenhum valor nulo
'''
# %% 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = '../data/creditcard.csv'
df = pd.read_csv(csv_path, sep=',')
# %%
# Mostra as 5 primeiras linhas do Dataset
df.head()
# %%

# Mostra métricas do Dataset
df.describe()

# %%
# Mostra informações gerais do Dataset
df.info()
# %%
# Soma por coluna os campos vazios
df.isna().sum()
# %%
# Soma no dataset os campos vazios
df.isnull().sum().max()
# %%
# Mosta o nome de cada coluna do Dataset
df.columns
# %%
no_fraud = df['Class'].value_counts()[0]
no_fraud_perc = round((no_fraud/len(df['Class']))*100,2)
fraud = df['Class'].value_counts()[1]
fraud_perc = round((fraud/len(df['Class']))*100,2)

# %%
print(f"Temos {no_fraud} uso legal que representa {no_fraud_perc}% de nosso Dataset")
print(f"Temos {fraud} uso fraudulento que representa {fraud_perc}% de nosso Dataset")

# %%
# Grádico a partir das quantidades 
sns.countplot(data=df, x='Class', hue='Class')
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
# %%
# Histograma dos valores para cada classe com curva de densidade estimada (kernel density estimation, KDE).
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.histplot(amount_val, ax=ax[0], color='r', kde=True)
ax[0].set_title('Distribuição Valor de Transação')
ax[0].set_xlim([min(amount_val), max(amount_val)])
ax[0].set_ylim(0, 10000)

sns.histplot(time_val, ax=ax[1], color='b', kde=True)
ax[1].set_title('Distribuição Tempo de Transação')
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()
# %%
# Identificando correlação
# Matriz Correlação
corr_matrix = df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm_r', fmt='.2f')
plt.title('Matriz Correlação')
plt.show()
# %%
#Pairplot
sns.pairplot(df)
plt.show()
# %%
# Preprocessamento
from sklearn.preprocessing import RobustScaler
# Inicializa o RobustScaler
rob_scaler = RobustScaler()

# Cria um novo DataFrame para as colunas escalonadas
df_scaled = df.copy()

# Ajusta o scaler e transforma as colunas 'Amount' e 'Time'
df_scaled['scaled_amount'] = rob_scaler.fit_transform(df[['Amount']])
df_scaled['scaled_time'] = rob_scaler.fit_transform(df[['Time']])

# Remove as colunas originais 'Time' e 'Amount'
df_scaled.drop(['Time', 'Amount'], axis=1, inplace=True)

# %%
df_scaled.info()
# %%
# Aplicando Random Under-Sampling
majority_class = df_scaled[df_scaled['Class'] == 0]  # Classe majoritária (uso legal)
minority_class = df_scaled[df_scaled['Class'] == 1]  # Classe minoritária (fraude)

minority_size = len(minority_class)

# Reduz aleatoriamente a classe majoritária para o mesmo tamanho da classe minoritária
majority_downsampled = majority_class.sample(n=minority_size, random_state=42)

# Combina as classes reduzidas
df_undersampled = pd.concat([majority_downsampled, minority_class])

# Embaralha o dataset para evitar viés
df_undersampled = df_undersampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Agora df_undersampled está balanceado
print(df_undersampled['Class'].value_counts())
# %%
# Matriz Correlação para as colunas escalonadas e balanceada
corr_matrix = df_undersampled.corr()

plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm_r', fmt='.2f')
plt.title('Matriz Correlação')
plt.show()
# %%
#Correlação positiva em: V4, V11, V2, V19
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V4', 
            data=df_undersampled,
            ax=axes[0])
axes[0].set_title('V4 vs Class Positive Correlation')
sns.boxplot(x='Class', y='V11', 
            data=df_undersampled,
            ax=axes[1])
axes[1].set_title('V11 vs Class Positive Correlation')
sns.boxplot(x='Class', y='V2', 
            data=df_undersampled,
            ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')
sns.boxplot(x='Class', y='V19', 
            data=df_undersampled,
            ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')
plt.show()

#Correlação negativa em: V14, V12, V10, V9
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.boxplot(x='Class', y='V14', 
            data=df_undersampled,
            ax=axes[0])
axes[0].set_title('V14 vs Class Negative Correlation')
sns.boxplot(x='Class', y='V12', 
            data=df_undersampled,
            ax=axes[1])
axes[1].set_title('V12 vs Class Negative Correlation')
sns.boxplot(x='Class', y='V10', 
            data=df_undersampled,
            ax=axes[2])
axes[2].set_title('V10 vs Class Negative Correlation')
sns.boxplot(x='Class', y='V9', 
            data=df_undersampled,
            ax=axes[3])
axes[3].set_title('V9 vs Class Negative Correlation')
plt.show()
# %%
