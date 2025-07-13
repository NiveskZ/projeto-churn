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
best_features = feature_importance[feature_importance['acumulada'] < 0.96]['index'].tolist()
best_features
# %%
# Modify
from feature_engine import discretisation, encoding
from sklearn import pipeline

# Modifica variáveis numéricas em valores discretos
# Nesse modelo utilizamos a árvore de decisão para fazer a discretização
tree_discretization = discretisation.DecisionTreeDiscretiser(variables=best_features,
                                                             regression=False,
                                                             bin_output='bin_number', # Transforma cada nó da árvore em um bin
                                                             cv=3
                                                             )

# OneHot
onehot= encoding.OneHotEncoder(variables=best_features, ignore_format=True)

# %%

"""arvore_nova = tree.DecisionTreeClassifier(random_state=42)
arvore_nova.fit(X_train_transform,y_train)

(pd.Series(arvore_nova.feature_importances_, index=X_train_transform.columns)
 .sort_values(ascending=False))"""
#%%
# MODEL
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import ensemble

# model = linear_model.LogisticRegression(penalty=None, random_state=42)
# model = naive_bayes.BernoulliNB()
#model = ensemble.RandomForestClassifier(random_state=42,
#                                        min_samples_leaf=20,
#                                        n_jobs=4,
#                                        n_estimators=1000
#                                        )

model = ensemble.AdaBoostClassifier(random_state=42,
                                    n_estimators=500,
                                    learning_rate=0.01)

model_pipeline = pipeline.Pipeline(
    steps=[('Discretizar',tree_discretization),
           ('OneHot', onehot),
           ('Model',model)
           ]
)

import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')

mlflow.set_experiment(experiment_name='churn_exp')

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train, y_train)

    from sklearn import metrics
    # Previsão e métricas base de treino
    y_train_predict = model_pipeline.predict(X_train)
    y_train_proba = model_pipeline.predict_proba(X_train)[:,1]

    acc_train = metrics.accuracy_score(y_train,y_train_predict)
    auc_train = metrics.roc_auc_score(y_train,y_train_proba)
    roc_train = metrics.roc_curve(y_train,y_train_proba)
    print("Acurácia Treino:", acc_train)
    print("AUC Treino:", auc_train)

    # Previsão base de teste
    #X_test_transform = tree_discretization.transform(X_test[best_features])
    #X_test_transform = onehot.transform(X_test_transform)

    y_test_predict = model_pipeline.predict(X_test)
    y_test_proba = model_pipeline.predict_proba(X_test)[:,1]

    acc_test = metrics.accuracy_score(y_test,y_test_predict)
    auc_test = metrics.roc_auc_score(y_test,y_test_proba)
    roc_teste = metrics.roc_curve(y_test,y_test_proba)
    print("Acurácia Treino:", acc_test)
    print("AUC Treino:", auc_test)

    # Previsão out of time
    #out_of_time_transform = tree_discretization.transform(out_of_time[best_features])
    #out_of_time_transform = onehot.transform(out_of_time_transform)

    y_oot_predict = model_pipeline.predict(out_of_time[features])
    y_oot_proba = model_pipeline.predict_proba(out_of_time[features])[:,1]

    acc_oot = metrics.accuracy_score(out_of_time[target],y_oot_predict)
    auc_oot = metrics.roc_auc_score(out_of_time[target],y_oot_proba)
    roc_oot = metrics.roc_curve(out_of_time[target],y_oot_proba)
    print("Acurácia oot:", acc_oot)
    print("AUC oot:", auc_oot)

    mlflow.log_metrics({
        "acc_train":acc_train,
        "auc_train":auc_train,
        "acc_test":acc_test,
        "auc_test":auc_test,
        "acc_oot": acc_oot,
        "auc_oot": auc_oot
    })
# %%
# Plotando curva ROC
plt.plot(roc_train[0],roc_train[1])
plt.plot(roc_teste[0],roc_teste[1])
plt.plot(roc_oot[0],roc_oot[1])
plt.grid()
plt.title("Curva ROC")
plt.legend([
    f"Treino: {100*auc_train:.2f}",
    f"Teste: {100*auc_test:.2f}",
    f"Out-Of-Time: {100*auc_oot:.2f}"
])
# %%
