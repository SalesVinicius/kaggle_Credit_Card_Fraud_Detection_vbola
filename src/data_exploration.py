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
# %%
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
