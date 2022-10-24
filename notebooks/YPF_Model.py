# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# <div style="padding:20px;color:#6A79BA;margin:0;font-size:180%;text-align:center;display:fill;border-radius:5px;background-color:white;overflow:hidden;font-weight:600">Inteligencia Artificial para la Predicción de Incremental de Presiones por Interferencia</div>
#
# <img src="https://metadata.fundacionsadosky.org.ar/media/media/images/YPF_-_sitio_WEB.png" style="border-radius:5px">
#
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">1 | Introducción</div>
#
# Una de las características asociadas a la producción de petróleo y gas no convencional es la necesidad de estimular los pozos productores antes del comienzo de la producción. La estimulación consiste en la generación de fisuras en la roca productora a través de la inyección de un fluido a alta presión.
#
# El avance sistemático de los desarrollos no convencionales en la Fm. Vaca Muerta implica la perforación de pozos horizontales distribuidos espacialmente con el objetivo de maximizar la recuperación de petróleo y gas. Con ese objetivo, la distancia entre pozos vecino ha ido disminuyendo a lo largo del tiempo y, como consecuencia, se han registrado interferencias entre los pozos en producción (pozos padres) y los pozos nuevos en estimulación (pozos hijos) denominados frac-hits.
#
# Los frac-hits consisten en una anomalía de presión, corte de agua y/o temperatura en un pozo productor vecino a un pozo que está siendo fracturado. Este suceso de denomina "golpe de fractura" o "frac-hit" debido a que las fracturas hidráulicas del pozo hijo "golpean" el volumen de reservorio estimulado por el pozo padre y generan la interconexión.
#
# Este fenómeno, se ha convertido en una importante preocupación debido a que constituye un riesgo para las operaciones, debido a las condiciones de integridad del pozo padre como colapso de casing, fugas de fluido por el cabezal de pozo, aprisionamiento de instalaciones de producción.
#
# La pérdida de integridad de un pozo padre trae aparejado un importante costo debido a los riesgos de seguridad que conlleva, la posible detención de sets de fractura, costos de remediación y costo de oportunidad de equipos que se destinan a asegurar pozos padres. Por lo tanto, se requiere desarrollar un modelo predictivo que permita identificar los pozos con riesgo a ser interferidos y la magnitud de los frac-hits asociados.
#
# En este contexto, el objetivo del concurso actual es el desarrollo de un algoritmo de predicción de incremento de presión como consecuencia de todos los frac-hits que pueda recibir un pozo padre a raíz de la estimulación de un conjunto de pozos hijos cercanos.
#
# Se busca optimizar los pozos que ingresan al protocolo de aseguramiento, minimizando dos tipos de pozos: los pozos no asegurados que son interferidos y los pozos asegurados que no son interferidos. El resultado mejora la gestión de los riesgos que se asume y optimizando los costos de lifting asociados. El alcance del modelo es para pozos productores en Yacimientos No Convencionales de Petróleo y Gas.
#
#
# Para mas detalles sobre los files y la descripción del problema ir a este [link](https://metadata.fundacionsadosky.org.ar/competition/29/)

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">2 | Enfoque de la solución</div>
#
# La variable target tiene
#
# Desde un principio, haciendo un leve análisis de los datos y con consultas en el workshop, se detectó que los datasets de train y validación están spliteados al azar sin considerar temporalidad, esto nos dejo abierta la puerta para plantear un modelo baseline tomando el primer delta futuro conocido.
#
# El modelo utilizado es [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/) que es un algoritmo de gradient boosting muy utilizado para este tipo de problemas.
#
# A medida que vayamos transitando la notebook la idea es ir detallando un poco mas cada parte del código ejecutado. Se han dejado comentadas parte de las pruebas realizadas sin mejoras.
#
#

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">3 | Obteniendo Data</div>
#

# + id="NjKshGeHZXPt"
from sys import displayhook

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_squared_error

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

USE_PRECOMPUTED = True
TUNE_PARAMETERS = False

# + id="xnA0ej_qZcuU"
root_folder = '../'
submission_folder = root_folder + 'submissions/'
data_folder = root_folder + 'data/'
version = 'v0.1'

train_df = pd.read_csv(
    data_folder + 'Dataset participantes.csv',
    encoding='utf-16le',
    sep='\t',
    decimal='.',
)
train_df['type'] = 'Train'
eval_df = pd.read_csv(data_folder + 'Dataset evaluación.csv')
eval_df['type'] = 'Test'
eval_df['delta_WHP'] = None
eval_df['WHP_i'] = eval_df['WHP_i'].replace(',', '.', regex=True).astype(float)
eval_df['D3D'] = eval_df['D3D'].replace(',', '.', regex=True).astype(float)

df = pd.concat([train_df, eval_df])
df = df.reset_index(drop=True)
df['delta_WHP'] = df['delta_WHP'].astype(float)

# -

df = df.sort_values(
    by=['FLUIDO', 'CAMPO', 'PAD_HIJO', 'PADRE', 'HIJO', 'ETAPA_HIJO']
)  # ,ascending=[True,True,True,True,True,False])

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">4 | EDA</div>

train_df.shape

displayhook(df.head())

# ### Variable target

displayhook(train_df['delta_WHP'].describe())
px.box(train_df['delta_WHP'], width=400, height=400)

# Observamos varias cosas:
# - que la variable tiene como valor 0 predominantemente,
# - la media en la data de train para todo el conjunto es una variación de 1.376
# - hay tanto incrementos como decrementos de presión

print(train_df.groupby(train_df['delta_WHP'] == 0).size() / train_df.shape[0] * 100)

# > En concreto el 88% de los registros son 0

# Armemos un modelo baseline con todos los registros en 0

baseline0 = df.loc[df.type == 'Test', ['ID_FILA', 'delta_WHP']]
baseline0['delta_WHP'] = 0.0
baseline0.to_csv(submission_folder + 'baseline0.csv', index=False, header=False)


# > Este modelo da un RSE de 1.23481

# +
def mse_ceros(df):
    print("MSE para ceros:")
    print(mean_squared_error(df.loc[df.type == 'Train', 'delta_WHP'], np.zeros(26178)))


mse_ceros(df)
# -

# Distribución de deltas distintos de cero

condition = train_df['delta_WHP'] != 0
displayhook(train_df.loc[condition, 'delta_WHP'].describe())
px.box(train_df.loc[condition, 'delta_WHP'], width=400, height=400)

# > Dentro de las estimulaciones que generaron deltas de presión, podemos hallar en un rango esperado a deltas entre -15.3 y 30.2, por fuera de esto ya tenemos outliers de variación

condition = (train_df['delta_WHP'].abs() < 1) & (train_df['delta_WHP'].abs() != 0)
displayhook(train_df.loc[condition, 'delta_WHP'].describe())

# > 355 casos con un delta menor a 1 en valor absoluto

# Valor absoluto de variaciones (sin considerar si es incremento o decremento)

train_df['abs_delta_WHP'] = abs(train_df['delta_WHP'])
round(train_df['abs_delta_WHP'].mean(), 3)

# ### Campo vs Fluido

df.groupby(['CAMPO', 'FLUIDO']).agg(
    {'PADRE': 'nunique', 'HIJO': 'nunique', 'type': ['min', 'max']}
)

# > Cada campo tiene destinado un sólo tipo de fluido

# > En todos los campos tengo data a inferir

# ### Variaciones de presión evaluado por zonas

# Según el workshop hay una ubicación espacial de los pozos dadas por el campo y el pad, veamos si encontramos particularidades en estas zonas

aux = (
    train_df.groupby(['CAMPO', 'PAD_HIJO'])['delta_WHP']
    .agg(['mean', 'count'])
    .reset_index()
)
print('Cantidad de Pads:')
displayhook(aux.shape[0])
condition = (aux['mean'] > 1.376) & (aux['count'] > 30)
print(
    'Pads con variaciones mayor a la media (>1.376) y cantidad de casos significativos (>30):',
    aux[condition].shape[0],
)
displayhook(aux[condition])

# Un segundo baseline podria ser completar para estos pads con la media

# +
mse_ceros(df)

aux2 = df.loc[df.type == 'Train', ['ID_FILA', 'delta_WHP']]
aux2['delta_WHP'] = 0.0
aux2['delta_WHP'] = (
    df[(df.type == 'Train') & (df.PAD_HIJO.isin(aux[condition].PAD_HIJO))]
    .groupby(['CAMPO', 'PAD_HIJO'], sort=False)['delta_WHP']
    .transform(lambda x: x.mean())
)
aux2['delta_WHP'] = aux2['delta_WHP'].fillna(0)
aux2['original_delta'] = df.loc[df.type == 'Train', 'delta_WHP']
print("\nMSE medias por Pad:")
displayhook(mean_squared_error(aux2['original_delta'], aux2['delta_WHP']))

aux2.to_csv(submission_folder + 'baseline_meanpad_train.csv', index=False, header=False)
# -

baseline1 = df.loc[df.type == 'Test', ['ID_FILA', 'delta_WHP']]
baseline1['delta_WHP'] = (
    df[df.PAD_HIJO.isin(aux[condition].PAD_HIJO)]
    .groupby(['CAMPO', 'PAD_HIJO'], sort=False)['delta_WHP']
    .transform(lambda x: x.mean())
)
baseline1['delta_WHP'] = baseline1['delta_WHP'].fillna(0.0)
baseline1.to_csv(submission_folder + 'baseline_meanpad.csv', index=False, header=False)

# > Este baseline empeora respecto del anterior, da 1.46569

# ### Padres vs Hijos

# N Interacciones de pozos hijos sobre pozos padres

aux = df.groupby('PADRE').HIJO.nunique().sort_values(ascending=False)
displayhook(aux.describe())
px.box(aux, width=400, height=400, title="n hijos por padre")

# ### Data Leakeage

# Los eventos tienen un orden temporal, si bien en este dataset no tenemos información de este tipo, tenemos las etapas de perforación de cada pozo, que si marcan un orden.
#
# Por otro lado sabemos que la data de evalución se armó aleatoriamente, es decir sin tener en cuenta las etapas, por lo cual podríamos tener data futura de como esta el pozo de relevancia para la competencia (obviamente esta data no se podría usar en un modelo productivo).
#
# Veamos si esta data aporta valor significativo

df.loc[
    df.CAMPO == 'Campo D',
    [
        'FLUIDO',
        'CAMPO',
        'PAD_HIJO',
        'PADRE',
        'HIJO',
        'ETAPA_HIJO',
        'type',
        'delta_WHP',
        'WHP_i',
    ],
].head(10)

# > Se observa que los casos a predecir estan de forma aleatoria entre las etapas

group_cols = ['FLUIDO', 'CAMPO', 'PAD_HIJO', 'PADRE', 'HIJO']
# calc shifted columns per padre-hijo relation
df['next_initial_pressure'] = df.groupby(group_cols)['WHP_i'].transform(
    lambda x: x.shift(-1)
)
df['next_stage'] = df.groupby(group_cols)['ETAPA_HIJO'].transform(lambda x: x.shift(-1))
df['prev_delta'] = df.groupby(group_cols)['delta_WHP'].transform(lambda x: x.shift(1))
df['prev_type'] = df.groupby(group_cols)['type'].transform(lambda x: x.shift(1))
df['estimated_delta'] = round(df['next_initial_pressure'] - df['WHP_i'], 3)
df['diff_stages'] = df['next_stage'] - df['ETAPA_HIJO']

# > Se calcula un delta estimado en funcion al valor inicial de presión de la próxima etapa

# A continuación agregaremos algunos filtros sobre esta estimación, como ser que sean etapas consecutivas, casos extremos o variaciones muy pequeñas:

# +
mse_ceros(df)

# if there is more than one stage between rows put delta 0
df.loc[(df['diff_stages'] != 1), 'estimated_delta'] = 0.0
# if delta is lower than 1 consider as noise
df.loc[(df['estimated_delta'].abs() < 1), 'estimated_delta'] = 0.0
# if delta is bigger than common delta values put fence value
df.loc[(df['estimated_delta'] > 30), 'estimated_delta'] = 30.0
df.loc[(df['estimated_delta'] < -15), 'estimated_delta'] = -15.0
# if prev delta value is 0, put 0 value
df.loc[
    ((df['prev_delta'] == 0.0) | (df['prev_delta'].isna())) & (df.prev_type != 'Test'),
    'estimated_delta',
] = 0.0
df.estimated_delta = df.estimated_delta.fillna(0)

print("\nMSE para estimacion:")
print(
    mean_squared_error(
        df.loc[df.type == 'Train', 'delta_WHP'],
        df.loc[df.type == 'Train', 'estimated_delta'],
    )
)

# -

# Veamos el MSE por pozo padre para los peores casos

aux = (
    df[(df.type == 'Train')]
    .groupby(['CAMPO', 'PAD_HIJO', 'PADRE'])
    .apply(lambda x: mean_squared_error(x['delta_WHP'], x['estimated_delta']))
)
aux[aux > 59]

df.loc[
    (df.PADRE == 'Pozo 29') & (df.HIJO == 'Pozo 466'),
    [
        'CAMPO',
        'PAD_HIJO',
        'PADRE',
        'HIJO',
        'ETAPA_HIJO',
        'type',
        'WHP_i',
        'next_initial_pressure',
        'delta_WHP',
        'estimated_delta',
        'diff_stages',
        'prev_delta',
    ],
]

pozo_padre = 'Pozo 29'
aux = df.loc[
    df.PADRE == pozo_padre,
    [
        'CAMPO',
        'PAD_HIJO',
        'PADRE',
        'HIJO',
        'ETAPA_HIJO',
        'type',
        'WHP_i',
        'next_initial_pressure',
        'delta_WHP',
        'estimated_delta',
        'diff_stages',
        'prev_delta',
    ],
]
fig = px.line(
    aux,
    x="ETAPA_HIJO",
    y="WHP_i",
    color='HIJO',
    hover_data=['type', 'delta_WHP', 'estimated_delta', 'diff_stages'],
    width=800,
    height=300,
)
fig.show()

fig = px.line(
    df[(df.PADRE == 'Pozo 29') & (df.HIJO == 'Pozo 466')],
    x="ETAPA_HIJO",
    y="delta_WHP",
    width=800,
    height=300,
)
fig.show()

# Creemos el baseline

baseline2 = df.loc[df.type == 'Test', ['ID_FILA', 'estimated_delta']]
baseline2.to_csv(
    submission_folder + 'baseline_estimated.csv', index=False, header=False
)


# > El score de este submit da 1.4157 no estaríamos mejorando

# + [markdown] id="0KrJZZAqZmlY"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">4 | Funciones</div>
#

# + [markdown] id="k9JH73RcZ8Va"
# ### Feature Enginneering
# -

# A continuacion encontraremos la función principal, su input los files originales, su output el dataset con las variables nuevas a utilizar por el modelo.

# + id="SIUsK0D5aDOn"
def fe(df):
    """Funcion principal de feature engineering. Recibe los sets de datos originales y devuelve un set de datos
    con las variables calculadas para realizar el entrenamiento.

    Args:
        df (pd.DataFrame): set de datos

    Returns:
        pd.DataFrame: set de datos final
    """

    return df


# -


def validate_model(df, target_col, feat_cols, params):
    """Entrenamiento del modelo contra set de validación

    Args:
        df (pd.DataFrame): set de datos completo
        target_col (str): nombre de la columna target
        feat_cols (list): lista de features del modelo
        params (dict): parametros del modelo

    Returns:
        Booster, array: modelo, lista de predicciones
    """
    train_filter = df['SetType'] == 'Train'
    valid_filter = df['SetType'] == 'Validation'
    list_str_obj_cols = df[feat_cols].columns[df[feat_cols].dtypes == "object"].tolist()
    for str_obj_col in list_str_obj_cols:
        df[str_obj_col] = df[str_obj_col].astype("category")
    print(
        f'Categorical columns:{df[feat_cols].columns[df[feat_cols].dtypes == "category"]}'
    )

    X_train, y_train = df.loc[train_filter, feat_cols], df.loc[train_filter, target_col]
    X_valid, y_valid = df.loc[valid_filter, feat_cols], df.loc[valid_filter, target_col]

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    lgb_results = {}
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=['Train', 'Valid'],
        verbose_eval=10,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals_result=lgb_results,
    )
    print(gbm.best_iteration)
    y_pred = gbm.predict(X_valid)
    mae = mean_squared_error(y_valid, y_pred)
    print("Best MAE:", mae)

    feat_importance = pd.DataFrame()
    feat_importance["Importance"] = gbm.feature_importance()
    feat_importance.set_index(X_train.columns, inplace=True)

    ################################
    # training chart
    plt.figure(figsize=(15, 10))

    # loss
    plt.subplot(1, 2, 1)
    loss_train = lgb_results['Train']['l1']
    loss_test = lgb_results['Valid']['l1']
    plt.xlabel('Iteration')
    plt.ylabel('mae')
    plt.plot(loss_train, label='train mae')
    plt.plot(loss_test, label='valid mae')
    plt.legend()

    # feature importance
    plt.subplot(1, 2, 2)
    importance = pd.DataFrame(
        {'feature': feat_cols, 'importance': gbm.feature_importance()}
    )
    sns.barplot(
        x='importance',
        y='feature',
        data=importance.sort_values('importance', ascending=False),
    )

    plt.tight_layout()
    plt.show()

    return gbm, y_pred


# + [markdown] id="k72M5nnqa4sp"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">5 | Validación Modelo</div>
#

# + [markdown] id="mvZECIc_dHYv"
# ### Features

# + id="KhHfwjnkdQQK"
header_cols = []
model_features = []
target = ''
# -

# ### Optimización parámetros

# Proceso de optimización de hiperparámetros del algortimo, puesto en false por defecto ya que es un poco lento (no mas de 10 min), de aquí salieron los parámetros seteados a continuación.

# if TUNE_PARAMETERS:
#     from verstack import LGBMTuner

#     train_filter = df['SetType'] == 'Train'
#     valid_filter = df['SetType'] == 'Validation'
#     list_str_obj_cols = (
#         df[model_features].columns[df[model_features].dtypes == "object"].tolist()
#     )
#     for str_obj_col in list_str_obj_cols:
#         df[str_obj_col] = df[str_obj_col].astype("category")
#     print(
#         f'Categorical columns:{df[model_features].columns[df[model_features].dtypes == "category"]}'
#     )

#     X_train, y_train = (
#         df.loc[train_filter, model_features],
#         df.loc[train_filter, target],
#     )
#     X_valid, y_valid = (
#         df.loc[valid_filter, model_features],
#         df.loc[valid_filter, target],
#     )
#     # tune the hyperparameters and fit the optimized model
#     tuner = LGBMTuner(metric='mae')  # <- the only required argument
#     tuner.fit(X_train, y_train)
#     # check the optimization log in the console.
#     pred = tuner.predict(X_valid)
#     mae = mean_squared_error(y_valid, pred)
#     print("Best MAE:", mae)
#     print(tuner.best_params)

# ### Parámetros

params = {
    'learning_rate': 0.05,
    'num_leaves': 242,
    'colsample_bytree': 0.8184760893279794,
    'subsample': 0.5382071904887805,
    'verbosity': -1,
    'random_state': 42,
    'objective': 'regression',
    'metric': 'l1',
    'num_threads': 14,
    'min_sum_hessian_in_leaf': 0.006706207126441575,
    'reg_alpha': 5.3006159843660915e-08,
    'reg_lambda': 0.0008648312997620131,
    'n_estimators': 430,
}


# ### Validación

# ### Análisis mayores diferencias respecto del target

# ### Mejoras predicciones en función de resultados

# + [markdown] id="6dAirrmtp8aM"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">6 | Entrenamiento Modelo</div>

# + id="YgYe0pcOo2Mr"
# def train_final(df, num_rounds, feat_cols, target_col, params):
#     """Entrenamiento del modelo final.

#     Args:
#         df (pd.DataFrame): set de datos full
#         num_rounds (int): numero de iteraciones
#         feat_cols (list): lista de features del modelo
#         target_col (str): nombre del target
#         params (dict): parametros de entrenamiento

#     Returns:
#         Booster: model
#     """
#     gbm = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=num_rounds)
#     return gbm


# model_final = train_final(df[df.PaymentPrincipal_historical_max.notnull()],
#                    model.best_iteration,
#                    model_features,
#                    target,
#                    params)


# + [markdown] id="k2Tl_IEdqH3c"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">7 | Inferencia</div>

# + id="_c3jUbxYqKAg"
def predict(df, feat_cols, model):
    """Calculo de predicciones.

    Args:
        df (pd.DataFrame): set de datos full
        feat_cols (list): lista de features del modelo
        model (Booster): modelo para realizar predicciones

    Returns:
        pd.DataFrame: set de datos con algunas variables del loan y su prediccion final
    """
    to_infer_filter = True
    to_infer = df.loc[to_infer_filter, feat_cols]

    result = df.loc[to_infer_filter, ['']]
    result['target'] = model.predict(to_infer)
    return result


# submission = predict(df, model_features, model_final)
# assert payments_scoring.shape[0]==submission.shape[0]
# submission_final = submission.copy()
# -


# ### Fixes sobre file de submission

# ### Guardando solución

# + id="dhcNOaT6rAaA"
# submission[['PaymentCode', 'PaymentPrincipal']].to_csv(
#     project_folder + 'submission' + version + '.csv', header=False, index=False
# )
# submission_final[['PaymentCode', 'PaymentPrincipal']].to_csv(
#     project_folder + 'submission_final' + version + '.csv', header=False, index=False
# )
# -

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">8 | Próximos pasos</div>
#

# De tener mas tiempo hubiera seguido con:
# - Generar predicciones a través de un [modelo de regressión lineal](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
# - Generar modelos a partir de mismo algoritmo pero variando los intervalos de tiempo, entrenar con menos meses mas cercanos a la fecha
# - Probar distintas formas de ensamblarlos
#

#
