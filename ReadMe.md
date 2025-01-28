# Credit Card Fraud Detection

Projeto feito a partir do dataset "Credit Card Fraud Detection" Disponibilizado no [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/code?datasetId=310&sortBy=commentCount)

## Introdução

Primeiro contato com o Dataset foi feito no `data_exploration.py`. O objetivo é conhecer os dados com que estou trabalhando. Descobrir o que precisa ser feito de tratamento de dados e as principais features.

Resumo:

- Temos 284315 uso legal que representa 99.83% de nosso Dataset.
- Temos 492 uso fraudulento que representa 0.17% de nosso Dataset.
- E o valor médio do uso do cartão de crédito é de 88,34.
- Não temos nenhum valor nulo.

Pontos de Alerta:

- O dataset contém, em sua maioria, dados de transações não fraudulentas. Utilizá-lo na forma como está pode influenciar o modelo ao causar overfitting, uma vez que o modelo pode aprender a assumir que a maioria das transações não são fraudulentas, comprometendo sua capacidade de identificar fraudes.


// Excluir
Balanceamento de Classes: Para lidar com esse problema, você pode usar técnicas como oversampling (ex.: SMOTE) ou undersampling para equilibrar as classes no dataset.
Escolha de Métricas: Certifique-se de usar métricas como AUC-ROC, F1-Score ou Precisão/Recall, pois métricas tradicionais como acurácia podem ser enganosas em datasets desbalanceados.
//