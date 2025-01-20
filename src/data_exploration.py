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
# %%
csv_path = '../data/creditcard.csv'
df = pd.read_csv(csv_path, sep=',')

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
