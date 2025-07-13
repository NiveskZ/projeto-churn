# Projeto Churn de Usuários
Esse repositório foi criado com o objetivo de realizar o projeto de prever o Churn de usuários do canal [TéoMeWhy](https://www.twitch.tv/teomewhy) utilizando os dados disponibilizado pelo próprio Téo na plataforma [Kaggle](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn). 

Aqui eu irei atualizar com minhas próprias anotações, tendo como foco principal destrinchar todas as tarefas feitas durante o curso gratuito de [Machine Learning 2025](https://github.com/TeoMeWhy/machine-learning-2025), feito também pelo Téo e disponibilizado [Aqui](https://www.youtube.com/playlist?list=PLvlkVRRKOYFR6_LmNcJliicNan2TYeFO2), recriar por conta própria a aplicação do framework **SEMMA** e entender seus passos principais.

## Ferramentas utilizadas
Durante todo o projeto foi utilizado a linguangem de programação **Python** e as seguintes bibliotecas:
- [pandas](https://pandas.pydata.org/docs/user_guide/index.html): Para manipulação, exploração e análise dos dados.
- [scikit-learn](https://scikit-learn.org/stable/): Seleção de amostragem, uso de métricas estatísticas e modelos de previsão.
- [matplotlib](https://matplotlib.org/): plotagem de gráficos para análise de modelos.
- [feature-engine](https://feature-engine.trainindata.com/en/latest/): Transformação e Modificação de variáveis

## Framework SEMMA
SEMMA é um framework criado pela SAS que significa:

- **S**ample
- **E**xplore
- **M**odify
- **M**odel
- **A**ssess

![SEMMA by SAS](https://miro.medium.com/v2/resize:fit:1324/0*o3UBmEz_3g6iDptz.JPG)

### Sample
Trabalhar com uma amostra representativa para que seja possível testar e validar o modelo.

Dentre as ações feitas nessa etapa podemos citar:

- Separar em amostras de treino e teste.
- Out of time
    - Criar uma terceira base, onde será verificada a estabilidade do modelo.
    - Geralmente são as últimas safras, pega-se a safra inteira.
    - Normalmente é feito quando se tem uma grande quantidade de dados.
- Fazer Balanceamento de base.
    - Modificar sinteticamente as proporções treino e teste na variável resposta.
    - Existem duas formas de fazer esse balanceamento: undersampling e oversampling
- Aplicar filtros.

### Explore
Essa etapa é onde fazemos a Análise Exploratória de Dados (EDA), daqui para frente utilizamos apenas a base de treino.

Através do nosso caso de exemplo, podemos observar por exemplo que os usuários mais engajados possuem menos chances de dar churn. Essa análise pôde ser feita através da tabela criada no seguinte código:
```py
df_analise = X_train
df_analise[target] = y_train
df_analise.groupby(by=target).agg(["mean","median"]).T 
```
Em que as variáveis relacionadas a engajamento possui uma média maior de "zeros". 

Portanto, na prática, podemos entender que fazer uma EDA é simplesmente olhar para a distribuição das covariáveis dentro da nossa variável resposta.

Em resumo, estamos conhecendo o dado através de:
- Análise descritiva.
- Análise Bivariada.
- Identificação de Missings.

### Modify
Agora é onde fazemos as modificações das colunas do nosso dataset, seja criar colunas novas sendo transformação das antigas, seja transformar algum tipo de coluna antiga em um valor novo. Alguns exemplos:

- Padronização: Colocar todas as variáveis na mesma escala.
- Imputação de Missing: Observar dados faltantes e entender como completa-lo ou tratá-lo, a depender do caso.
- Binning: Transformar variáveis numéricas em ordinal ou nominal.
- Combinação

### Model
Aplicação dos resultados obtidos até aqui em algum modelo de predição ou estátistico:

- Modelos Estatísticos: Cálculo de métricas (Acurácia, curva ROC, etc)
- Modelos de Árvore: Árvores de Decisão, classificação, Regressão, etc.
- Modelos de Vetor de Suporte.
- Redes Neurais.

### Assess
Essa é a última etapa, onde avaliamos as métricas de acurácia do nosso modelo e sua utilidade. Aqui fazemos toques finais, podendo incluir:
- Métricas de Ajuste
- Decisão
- Comparação
- Serialização