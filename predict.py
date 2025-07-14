# %%
import mlflow.sklearn
import pandas as pd
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Import do Modelo

model_df = pd.read_pickle("model.pkl")
model = model_df['model']
features = model_df['features']

# Modelo mlflow
model_mlflow = mlflow.sklearn.load_model("models:/model_churn/1")
# %%
model_mlflow
features_mlflow = model_mlflow.feature_names_in_
features_mlflow
#%%
# Import de "Novos" dados
df = pd.read_csv("data/abt_churn.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)
amostra = amostra.drop('flagChurn',axis=1)
# %%

# Predição
predicao = model.predict_proba(amostra[features])[:,1]
predicao
# %%
