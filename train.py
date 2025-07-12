# %%
import pandas as pd

df = pd.read_csv('data/abt_churn.csv')
df.head()

# %%
# Selecionando o Out of time, Safra por mês.
out_of_time = df[df['dtRef'] == df['dtRef'].max()].copy()
out_of_time
# %%
df_train = df[df['dtRef'] < df['dtRef'].max()].copy()
df_train['dtRef']
# %%
df_train.head()
# %%
# Variáveis
features = df_train.columns[2:-1]

# Variável target que queremos prever
target = 'flagChurn'

X,y = df_train[features], df_train[target]
# %%
from sklearn import model_selection

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,
                                                                 random_state=42,
                                                                 test_size=0.2,
                                                                 stratify=y # Garante que as duas amostras tenha a mesma taxa da variável resposta (Seleciona 20% de todos os 0 para teste e igualmente 20% de todos os 1)
                                                                 )
# %%
# Verificando se as amostras são parecidas (representatividade das amostras)
print("Taxa variável resposta Treino:", y_train.mean())
print("Taxa variável resposta Teste:",y_test.mean())

# Resultado sem estratificação (stratify) = Taxa variável resposta Treino: 0.4715936446798267
# Resultado sem estratificação (stratify) = Taxa variável resposta Teste: 0.45813282001924927
# ---------------------------------------------------------------------------------------------
# Resultado com estratificação = Taxa variável resposta Treino: 0.46894559460760715
# Resultado com estratificação = Taxa variável resposta Teste: 0.4687199230028874
# %%
