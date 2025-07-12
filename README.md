# Projeto Churn de Usuários
Esse repositório foi criado com o objetivo de realizar o projeto de prever o Churn de usuários do canal [TéoMeWhy](https://www.twitch.tv/teomewhy) utilizando os dados disponibilizado pelo próprio Téo na plataforma [Kaggle](https://www.kaggle.com/datasets/teocalvo/analytical-base-table-churn). 

Aqui eu irei atualizar com minhas próprias anotações, tendo como foco principal destrinchar todas as tarefas feitas durante o curso gratuito de [Machine Learning 2025](https://github.com/TeoMeWhy/machine-learning-2025), feito também pelo Téo e disponibilizado [Aqui](https://www.youtube.com/playlist?list=PLvlkVRRKOYFR6_LmNcJliicNan2TYeFO2), recriar por conta própria a aplicação do framework **SEMMA** e entender seus passos principais.

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