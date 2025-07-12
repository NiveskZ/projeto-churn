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
# SAMPLE
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
# EXPLORE
# Verificando dados faltantes.
X_train.isna().sum().sort_values(ascending=False)
# %%
df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(["mean","median"]).T

sumario
# %%
sumario['diff_abs'] = sumario[0] - sumario[1]
sumario['diff_rel'] = sumario[0] / sumario[1]
sumario.sort_values(by=['diff_rel'],ascending=False).head(20)
# %%

from sklearn import tree
import matplotlib.pyplot as plt

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train,y_train)

#plt.figure(dpi=400, figsize=[4,4])
#tree.plot_tree(arvore, feature_names=X_train.columns,
#               filled=True,
#               class_names=[str(i) for i in arvore.classes_])
# %%
# Associação de cada feature com sua "importância"

feature_importance = (pd.Series(arvore.feature_importances_, 
                               index=X_train.columns)
                               .sort_values(ascending=False)
                               .reset_index()
                               )
feature_importance['acumulada'] = feature_importance[0].cumsum()

# Fazendo uma feature selection
feature_importance[feature_importance['acumulada'] < 0.96]
# %%
