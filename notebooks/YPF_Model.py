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

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">2 | Evolución y enfoque final de la solución</div>
#
# En esta sección describiremos el proceso transitado durante la competencia y las sucesivas marchas y contramarchas dentro de la competencia.
#
# - Submit 1:
#
# Luego de un análisis exploratorio incial (pandas profiling no agregado a esta notebook + algunos distribuciones de la variable target si agregadas) se probo un baseline con todos los delta en 0 (1.21) para tener un score de referencia
#
# - Submit 2
#
# Teniendo a ese momento como métrica de referencia el MSE y estando en una etapa puramente exploratoria se busco la completitud por medias para los pads con mayor delta que la media del dataset, esto mejoraba el MSE pero empeoró el lb (1.46)
#
# - Submit 3
#
# Data leakeage, luego del workshop se aclaró q las etapas tienen un orden cronológico y que la data de evaluación se habia extraído de forma aleatoria, por lo cual se jugo con variables de etapas posteriores para estimar el delta, nuevamente mejoraba el MSE  pero no el lb (1.41), acá se valido que la métrica utilizada en el lb es el MAE y se empezó a utilizar para medir mejoras
#
# - Submit 4
#
# Se utilizó el primer modelo predictivo, se utilizó [LightGBM](https://lightgbm.readthedocs.io/en/v3.3.1/) que es un algoritmo de gradient boosting muy popular para este tipo de problemas. Sin ningun tipo de feature enginnering (de ahora en mas fe) adicional al que ya se tenía, spliteando 80/20 para tener un set de validación, y luego entrenando con la totalidad de la data y el número de iteraciones obtenida contra esa validación. Ahí ya utilizando MAE en validación daba 1.22 y en el lb se consiguió un puntaje de 0.90 obteniendo mejora significativa
#
# - Submits 5,6,7
#
# Se empezaron a crear funciones de fe, y esto trajo errores en el script. La función inicialmente no tenia el sort indicado sobre los registros para que el cálculo de las variables shifted diera de forma correcta por lo cual se obtuvieron dos submits fallidos.
#
# Una vez detectado y solucionado este error el modelo agregaba algunas variables de la etapa anterior y una variable que sale de obtener una distancia jugando con ángulos y distancias (ver función de variables trigonométricas), esto daba una sutil mejora en validación (1.21) pero empeoraba en el lb (0.94).
#
# - Submits 8,9,10
#
# Sin mejoras, se crearon variables relativas a la etapa (ver función stage_vars), se fixearon bugs en el script respecto de esas variables (problemas con train, validation y test flags), y se probo variable max_delta_last_3, mejora en validacion score (1.08) pero no en lb (0.93113)
#
# - Submit 11
#
# Jugando con variable ID_FILA, pensando erróneamente que podía aportar info del orden de los sucesos, empeoraba en validación (1.30) y tambien en lb (1.14).
#
# Mucho tiempo dedicado a tratar de reconstruir la linea temporal de un pozo padre (ordenar las interacciones de los pozos hijos) sin éxito.
#
# - Submit 12
#
# Abandonada la idea de rearmar el flujo de alteraciones se generan variables futuras que son de peso para la mejora como:
#     'max_delta_next_6',
#     'mean_delta_next_6',
#     'next_pressure_relation',
#
# Dando en validation 0.8264 y en lb 0.86 (septimo hasta el momento)
#

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">3 | Obteniendo Data</div>
#

import math

# + id="NjKshGeHZXPt"
from sys import displayhook

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

USE_PRECOMPUTED = True
TUNE_PARAMETERS = False

# + id="xnA0ej_qZcuU"
root_folder = '../'
submission_folder = root_folder + 'submissions/'
data_folder = root_folder + 'data/'
version = 'v12'


def read_data():
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
    return train_df, eval_df, df.copy()


train_df, eval_df, df = read_data()

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

    print("MAE para ceros:")
    print(mean_absolute_error(df.loc[df.type == 'Train', 'delta_WHP'], np.zeros(26178)))


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

df.groupby('PADRE').size().sort_values().tail()

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
print("\nMAE medias por Pad:")
displayhook(mean_absolute_error(aux2['original_delta'], aux2['delta_WHP']))

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

# > Tengo una mediana de 6 hijos interfiriendo un padre y el 50% de la distribución esta entre 3 y 10 hijos

# ### Data Leakeage

# Los eventos tienen un orden temporal, si bien en este dataset no hay fechas, tenemos las etapas de perforación de cada pozo, que si marcan un orden.
#
# Por otro lado sabemos que la data de evalución se armó aleatoriamente, es decir sin tener en cuenta las etapas, por lo cual podríamos tener data futura de como esta el pozo de relevancia para la competencia (obviamente esta data no se podría usar en un modelo productivo).
#
# Veamos si esta data aporta valor significativo

df.loc[
    df.CAMPO == 'Campo D',
    [
        'ID_FILA',
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

# > Se observa que los casos a predecir estan de forma aleatoria entre las etapas, pero también vemos que en los casos de train, la variable ID_FILA pareciera identificar el orden de los casos

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
upper_fence = 13.3
lower_fence = 0
df.loc[(df['estimated_delta'] > upper_fence), 'estimated_delta'] = upper_fence
df.loc[(df['estimated_delta'] < lower_fence), 'estimated_delta'] = lower_fence
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
print("\nMAE para estimacion:")
print(
    mean_absolute_error(
        df.loc[df.type == 'Train', 'delta_WHP'],
        df.loc[df.type == 'Train', 'estimated_delta'],
    )
)

# -

pozo_padre = 'Pozo 212'
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
    # width=800,
    # height=300,
    title="Presión inicial por etapa",
)
fig.show()

fig = px.line(
    df[(df.PADRE == pozo_padre)],  # & (df.HIJO == 'Pozo 466')],
    x="ETAPA_HIJO",
    y="delta_WHP",
    color="HIJO",
    width=800,
    height=300,
    title="Delta por etapa",
)
fig.show()

# Creemos el baseline

baseline2 = df.loc[df.type == 'Test', ['ID_FILA', 'estimated_delta']]
baseline2.to_csv(
    submission_folder + 'baseline_estimated.csv', index=False, header=False
)


# > El score de este submit da 1.4157 no estaríamos mejorando (hasta este momento solo estaba revisando contra MSE, de aquí en mas ya mi métrica será MAE)

# ### Análisis ID_FILA

# Como mencionabamos anteriormente, esta variable parece marcar un orden temporal, esto nos permitiria ordenar los eventos no solo en un pozo hijo (ya lo lograbamos con la etapa) sino tambien entre pozos que afecten un mismo padre


def chart_evolution(
    df,
    x='ID_FILA',
    y='WHP_i',
    color='HIJO',
    title='presion inicial por ID fila',
    father_name=None,
):
    if father_name is None:
        aux = df.sort_values(by=x)
        title = "Evolución " + title
    else:
        aux = df[(df.PADRE == father_name)].sort_values(by=x)
        title = "Evolución " + title + " del pozo padre: " + father_name
    fig = px.scatter(
        aux,
        x=x,
        y=y,
        color=color,
        hover_data=[
            'type',
            'delta_WHP',
            'estimated_delta',
            'CAMPO',
            'HIJO',
            'ETAPA_HIJO',
        ],
        # width=800,
        # height=300,
        title=title,
    )
    fig.show()


chart_evolution(df, color='CAMPO')

# > Evidentemente el set de train se ordeno por presión inicial en varios tramos pero esto no tiene nada que ver con el orden de los eventos, en un principio parecia ya que la presión inicial de los pozos suele incrementarse

# ### Evolución presión

title = 'presión inicial'
chart_evolution(
    df, x='ETAPA_HIJO', father_name='Pozo 169', title='presión inicial por etapa'
)
# chart_evolution(df, x='id_row', father_name='Pozo 169', title='presión inicial por nuevo orden')
chart_evolution(
    df, x='ETAPA_HIJO', father_name='Pozo 212', title='presión inicial por etapa'
)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 212', title=title)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 5', title=title)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 382', title=title)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 91', title=title)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 28', title=title)
chart_evolution(df, x='ETAPA_HIJO', father_name='Pozo 175', title=title)


# A continuación intentaremos generar una variable similar a ID_FILA pero que contemple los casos de evaluación

# +
# sin resultados
def smart_sort(df):
    fathers = [
        'Pozo 212'
    ]  # ,'Pozo 212','Pozo 91','Pozo 5','Pozo 382','Pozo 28','Pozo 175']
    df['PADRE_int'] = df.PADRE.str[5:].astype(int)
    df['HIJO_int'] = df.HIJO.str[5:].astype(int)
    df = df[df.PADRE.isin(fathers)].sort_values(
        ['FLUIDO', 'CAMPO', 'PADRE_int', 'HIJO_int', 'ETAPA_HIJO']
    )
    # df = df.sort_values(['FLUIDO','CAMPO','PADRE_int','HIJO_int','ETAPA_HIJO'])

    df['HIJO_group'] = df.HIJO_int.floordiv(10)
    # df = df.sort_values(['FLUIDO','CAMPO','PADRE_int','HIJO_group','ETAPA_HIJO'])
    df = df.drop(['id_row'], axis=1, errors='ignore')
    df.index.name = 'id_row'
    df = df.reset_index()

    # for father in list(aux.PADRE.unique()):
    #     print(father)
    #     n_sons = 1
    #     for son in list(aux.HIJO.unique()):
    #         print(son)
    #         if (n_sons == 1):
    #             result = pd.concat([result,aux[(aux.PADRE==father) & (aux.HIJO==son)]])
    #         else:
    #             for r in result.iterrows:
    #                 for s in aux[(aux.PADRE==father) & (aux.HIJO==son)]
    #         n_sons = n_sons + 1
    # display(result)
    # print(result.shape)
    return df


# df = smart_sort(df)


def estimation_by_id_row(df):
    df['id_row'] = df[df.type == 'Train'].ID_FILA
    group_cols = ['FLUIDO', 'CAMPO', 'PADRE']
    df = df.sort_values(by=group_cols + ['WHP_i'])
    grouped = df.groupby(group_cols)
    # df['id_row'] = df['id_row'].isna().cumsum() + df['days_unseen'].ffill()
    df['id_row'] = grouped['id_row'].transform(lambda x: x.interpolate())

    df = df.sort_values(by='id_row')
    grouped = df.groupby(group_cols)
    # calc shifted columns per padre-hijo relation
    df['next_initial_pressure'] = grouped['WHP_i'].transform(lambda x: x.shift(-1))
    df['prev_delta'] = grouped['delta_WHP'].transform(lambda x: x.shift(1))
    df['next_row'] = grouped['id_row'].transform(lambda x: x.shift(-1))

    df['max_delta_last_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=6, min_periods=1).max().shift()
    )
    df['mean_delta_last_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean().shift()
    )

    df['prev_type'] = grouped['type'].transform(lambda x: x.shift(1))
    df['estimated_delta'] = round(df['next_initial_pressure'] - df['WHP_i'], 3)
    df['diff_next_row'] = df['next_row'] - df['id_row']

    # df.loc[(df['estimated_delta'].abs() < 1), 'estimated_delta'] = 0.0
    # # if delta is bigger than common delta values put fence value
    # upper_fence = 13.3
    # lower_fence = 0
    # df.loc[(df['estimated_delta'] > upper_fence), 'estimated_delta'] = upper_fence
    # df.loc[(df['estimated_delta'] < lower_fence), 'estimated_delta'] = lower_fence
    # if last delta values are 0, put 0 value
    df.loc[
        (df['prev_delta'] == 0.0),
        'estimated_delta',
    ] = 0.0
    # df.estimated_delta = df.estimated_delta.fillna(0)
    return df


df = estimation_by_id_row(df)
# display(df.loc[(df.PADRE=='Pozo 212'),['id_row','HIJO','ETAPA_HIJO','WHP_i','delta_WHP','estimated_delta','type','next_initial_pressure','prev_delta','mean_delta_last_6','max_delta_last_6']])
chart_evolution(df, father_name='Pozo 169', x='id_row', color='type')
chart_evolution(df, father_name='Pozo 212', x='id_row', color='type')
chart_evolution(df, father_name='Pozo 5', x='id_row', color='type')
chart_evolution(df, father_name='Pozo 382', x='id_row', color='type')
chart_evolution(df, father_name='Pozo 91', x='id_row', color='type')


# + [markdown] id="0KrJZZAqZmlY"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">4 | Funciones</div>
#

# + [markdown] id="k9JH73RcZ8Va"
# ### Feature Enginneering
# -


def shifted_vars(df):
    group_cols = ['FLUIDO', 'CAMPO', 'PAD_HIJO', 'PADRE', 'HIJO']
    df = df.sort_values(by=group_cols)
    grouped = df.groupby(group_cols)
    # calc shifted columns per padre-hijo relation

    df['prev_delta'] = grouped['delta_WHP'].transform(lambda x: x.shift(1))
    df['prev_init_pressure'] = grouped['WHP_i'].transform(lambda x: x.shift(1))

    df['max_delta_last_3'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=3, min_periods=1).max().shift()
    )

    df['prev_stage'] = grouped['ETAPA_HIJO'].transform(lambda x: x.shift(1))
    # df['prev_type'] = grouped['type'].transform(lambda x: x.shift(1))
    # df['diff_prev_stage'] = df['ETAPA_HIJO'] - df['prev_stage']

    df['next_initial_pressure'] = grouped['WHP_i'].transform(lambda x: x.shift(-1))
    df['next_stage'] = grouped['ETAPA_HIJO'].transform(lambda x: x.shift(-1))
    df['diff_next_stage'] = df['next_stage'] - df['ETAPA_HIJO']
    df['estimated_delta'] = round(df['next_initial_pressure'] - df['WHP_i'], 3)

    df['max_delta_last_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=6, min_periods=1).max().shift()
    )
    df['mean_delta_last_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean().shift()
    )

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=6)
    df['max_delta_next_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=indexer, min_periods=1).max().shift()
    )
    df['mean_delta_next_6'] = grouped['delta_WHP'].transform(
        lambda x: x.rolling(window=indexer, min_periods=1).mean().shift()
    )
    df['prev_pressure_relation'] = df['WHP_i'] / df['prev_init_pressure']
    df['next_pressure_relation'] = df['next_initial_pressure'] / df['WHP_i']

    # if there is more than one stage between rows put delta 0
    df.loc[(df['diff_next_stage'] != 1), 'estimated_delta'] = 0.0
    # if delta is lower than 1 consider as noise
    df.loc[(df['estimated_delta'].abs() < 1), 'estimated_delta'] = 0.0
    # if delta is bigger than common delta values put fence value
    upper_fence = 13.3
    lower_fence = 0
    df.loc[(df['estimated_delta'] > upper_fence), 'estimated_delta'] = upper_fence
    df.loc[(df['estimated_delta'] < lower_fence), 'estimated_delta'] = lower_fence
    # if prev delta value is 0, put 0 value
    df.loc[
        (df['prev_delta'].isna()),
        'estimated_delta',
    ] = 0.0
    df.estimated_delta = df.estimated_delta.fillna(0)

    return df


# +
def trigonometric_vars(df):
    # equivalent to D2D
    # df['DZ_D3D_pithagoras'] = (df['D3D'].pow(2) - df['DZ'].pow(2)).pow(0.5)

    df.loc[df.AZ == 0, 'AZ_D2D_oposite'] = 0
    df.loc[df.AZ > 0, 'AZ_D2D_oposite'] = df.D2D * df.AZ.apply(
        lambda x: math.sin(math.radians(x))
    )
    df.loc[df.AZ > 90, 'AZ_D2D_oposite'] = df.D2D * df.AZ.apply(
        lambda x: math.sin(math.radians(180 - x))
    )
    df.loc[df.AZ > 180, 'AZ_D2D_oposite'] = df.D2D * df.AZ.apply(
        lambda x: math.sin(math.radians(x - 180))
    )
    df.loc[df.AZ > 270, 'AZ_D2D_oposite'] = df.D2D * df.AZ.apply(
        lambda x: math.sin(math.radians(360 - x))
    )

    return df


# trigonometric_vars(df.head(100))[['D3D','DZ','AZ','D2D','AZ_D2D_oposite']]


# +
def stage_vars(df):
    group_cols = ['HIJO', 'ETAPA_HIJO']
    grouped = df[df.type == 'Train'].groupby(group_cols)
    df['n_padres_in_stage'] = grouped['PADRE'].transform(lambda x: x.nunique())
    # df['max_delta_in_stage'] = grouped['delta_WHP'].transform(lambda x: x.max())
    # df['min_delta_in_stage'] = grouped['delta_WHP'].transform(lambda x: x.min())
    df['n_deltas_not_cero_in_stage'] = grouped['delta_WHP'].transform(
        lambda x: sum(abs(x) > 0)
    )

    return df


# condition = (df.HIJO=='Pozo 461') & (df.ETAPA_HIJO==7)
# #condition = df.delta_WHP>10
# #df.sample(10)
# display(df[condition])
# stage_vars(df[condition])


# -


def dad_vars(df):
    group_cols = ['FLUIDO', 'CAMPO', 'PAD_HIJO', 'PADRE']
    df = df.sort_values(by=group_cols + ['WHP_i'])
    grouped = df.groupby(group_cols)

    df['n_padres_in_stage'] = grouped['PADRE'].transform(lambda x: x.nunique())
    # df['max_delta_in_stage'] = grouped['delta_WHP'].transform(lambda x: x.max())
    # df['min_delta_in_stage'] = grouped['delta_WHP'].transform(lambda x: x.min())
    df['n_deltas_not_cero_in_stage'] = grouped['delta_WHP'].transform(
        lambda x: sum(abs(x) > 0)
    )

    return df


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
    df = df.sort_values(
        by=['FLUIDO', 'CAMPO', 'PAD_HIJO', 'PADRE', 'HIJO', 'ETAPA_HIJO']
    )  # ,ascending=[True,True,True,True,True,False])

    df['PADRE_int'] = df.PADRE.str[5:].astype(int)
    df['HIJO_int'] = df.HIJO.str[5:].astype(int)
    df['HIJO_group'] = df.HIJO_int.floordiv(10)

    df = shifted_vars(df)
    df = trigonometric_vars(df)
    df = stage_vars(df)
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
    train_filter = df['type'] == 'Train'
    valid_filter = df['type'] == 'Validation'
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
    mae = mean_absolute_error(y_valid, y_pred)
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


def lgbm_cross_validation(
    df,
    target_col,
    feat_cols,
    params,  # n_splits=3, random_state=2221
):

    list_str_obj_cols = df[feat_cols].columns[df[feat_cols].dtypes == "object"].tolist()
    for str_obj_col in list_str_obj_cols:
        df[str_obj_col] = df[str_obj_col].astype("category")

    cats_feats = df[feat_cols].columns[df[feat_cols].dtypes == "category"].values
    print(f'Categorical columns:{cats_feats}')

    lgb_train = lgb.Dataset(df[feat_cols], label=df[target_col])
    lgb.cv(params, lgb_train, categorical_feature=list(cats_feats))
    return None


# + [markdown] id="k72M5nnqa4sp"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">5 | Validación Modelo</div>
#
# -

# reload data
train_df, eval_df, df = read_data()


train, test = train_test_split(
    df[(df.type != 'Test')], test_size=0.3, random_state=1112
)
df.loc[test.index, 'type'] = 'Validation'
df = fe(df)

# + [markdown] id="mvZECIc_dHYv"
# ### Features

# + id="KhHfwjnkdQQK"
header_cols = ['ID_FILA', 'ID_EVENTO', 'type']
model_features = [
    'CAMPO',
    'FLUIDO',
    'PAD_HIJO',
    'HIJO',
    'ETAPA_HIJO',
    'PADRE',
    'D3D',
    'D2D',
    'DZ',
    'AZ',
    '#_BARRERAS',
    'LINEAMIENTO',
    'WHP_i',
    'ESTADO',
    'prev_delta',
    'HIJO_int',
    'PADRE_int',
    # 'prev_stage',
    # 'diff_prev_stage',
    'max_delta_last_6',
    'mean_delta_last_6',
    'prev_pressure_relation',
    'AZ_D2D_oposite',
    'max_delta_next_6',
    'mean_delta_next_6',
    'next_pressure_relation',
    'estimated_delta',
    # 'id_row',
    # 'diff_next_row',
    # 'n_padres_in_stage',
    # 'max_delta_in_stage',
    # 'min_delta_in_stage',
    # 'n_deltas_not_cero_in_stage'
]
target = 'delta_WHP'
# -

# ### Optimización parámetros

# Proceso de optimización de hiperparámetros del algortimo, puesto en false por defecto ya que es un poco lento (no mas de 10 min), de aquí salieron los parámetros seteados a continuación.

# +
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
#     mae = mean_absolute_error(y_valid, pred)
#     print("Best MAE:", mae)
#     print(tuner.best_params)
# -

# ### Parámetros

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.01,
    #    'lambda_l1': 0.5,
    #    'lambda_l2': 0.5,
    'num_leaves': 100,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'seed': 1003,
    'n_estimators': 2000,
}


# ### Validación

df.type.value_counts()

df.type.value_counts()

# +
# lgbm_cross_validation(df[(df.type != 'Test')].copy(), target, model_features, params, n_splits=3)
# -

model, preds = validate_model(df[(df.type != 'Test')], target, model_features, params)
# Best MAE: 1.07

# ### Análisis mayores diferencias respecto del target

# ### Mejoras predicciones en función de resultados

# + [markdown] id="6dAirrmtp8aM"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">6 | Entrenamiento Modelo</div>
# -


# reload data
train_df, eval_df, df = read_data()
df = fe(df)


# + id="YgYe0pcOo2Mr"
def train_final(df, num_rounds, feat_cols, target_col, params):
    """Entrenamiento del modelo final.

    Args:
        df (pd.DataFrame): set de datos full
        num_rounds (int): numero de iteraciones
        feat_cols (list): lista de features del modelo
        target_col (str): nombre del target
        params (dict): parametros de entrenamiento

    Returns:
        Booster: model
    """
    list_str_obj_cols = df.columns[df.dtypes == "object"].tolist()
    for str_obj_col in list_str_obj_cols:
        df[str_obj_col] = df[str_obj_col].astype("category")

    train_filter = df.type.isin(['Train', 'Validation'])
    X_train, y_train = df.loc[train_filter, feat_cols], df.loc[train_filter, target_col]
    gbm = lgb.train(params, lgb.Dataset(X_train, y_train), num_boost_round=num_rounds)
    return gbm


model_final = train_final(df, model.best_iteration, model_features, target, params)


# + [markdown] id="k2Tl_IEdqH3c"
# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">7 | Inferencia</div>

# + id="_c3jUbxYqKAg"
def predict(df, header_cols, feat_cols, target_col, model):
    """Calculo de predicciones.

    Args:
        df (pd.DataFrame): set de datos full
        feat_cols (list): lista de features del modelo
        model (Booster): modelo para realizar predicciones

    Returns:
        pd.DataFrame: set de datos con algunas variables del loan y su prediccion final
    """
    to_infer_filter = df.type == 'Test'
    to_infer = df.loc[to_infer_filter, feat_cols]

    result = df.loc[to_infer_filter, header_cols]
    result[target_col] = model.predict(to_infer)

    return result


submission = predict(df, header_cols, model_features, target, model_final)
submission.head()
# -


# ### Fixes sobre file de submission

# ### Guardando solución

# + id="dhcNOaT6rAaA"
submission[['ID_FILA', target]].to_csv(
    submission_folder + 'submit_' + version + '.csv', index=False, header=False
)

# -

# > Este modelo dio 0.86129 de score, 7 puesto hasta el momento

# # <div style="padding:20px;color:white;margin:0;font-size:100%;text-align:left;display:fill;border-radius:5px;background-color:#6A79BA;overflow:hidden">8 | Próximos pasos</div>
#

#
