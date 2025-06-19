# ===============================
# 1: Preparacion de entorno y cargue de librerias
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from statsmodels.tsa.statespace.sarimax import SARIMAX



from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 2: CARGA DEL DATASET - procesamiento y manejo del Date_time
# ===============================

# Cargar archivo  CSV
df = pd.read_csv("skechers_googleads.csv")

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

# Renombrar columnas claves para estandarizar el análisis
df = df.rename(columns={
    'día': 'fecha',  ## Preferible para evitar errores posteriores
    'campaña': 'campaña',
    'coste': 'costo',
    'impresiones': 'impresiones',
    'clics': 'clics',
    'cpc medio': 'cpc_medio', ## Codificacion adecuada para manipulacion
    'avg. cpm': 'cpm_medio',  ## Codificacion adecuada para manipulacion
    'ctr': 'ctr',
    'cpv medio': 'cpv_medio',  ## Codificacion adecuada para manipulacion
    'valor conv. total': 'valor_conversiones'  ## Codificacion adecuada para manipulacion
})

## Diccionario para traducir meses en español a numeros

meses_esp = {
    r'\bene\b': '01',
    r'\bfeb\b': '02',
    r'\bmar\b': '03',
    r'\babr\b': '04',
    r'\bmay\b': '05',
    r'\bjun\b': '06',
    r'\bjul\b': '07',
    r'\bago\b': '08',
    r'\bsep\b': '09',
    r'\bsept\b': '09',
    r'\boct\b': '10',
    r'\bnov\b': '11',
    r'\bdic\b': '12'
}

## Validacion de columna tipo texto

df['fecha'] = df['fecha'].astype(str).str.lower()

## Conversion de mes con expresiones regulares

for mes, num in meses_esp.items():
    df['fecha'] = df['fecha'].apply(lambda x: re.sub(mes, num, x))
    
## Conversion a datatime

df['fecha'] = pd.to_datetime(df['fecha'], format = '%d %m %Y')

## Confirmacion de ajuste en fechas exitosa

print("Conversion de fechas exitosa. Primeras fechas: ")
print(df['fecha'].head())

# ===============================
# 3: Agregacion de eventos calendario
# ===============================

from datetime import date

## Lista de eventos relevantes para la marca

eventos_skechers = [
    {"evento": "Día del Padre", "inicio": date(2021, 6, 20), "fin": date(2021, 6, 20)},
    {"evento": "Día del Padre", "inicio": date(2022, 6, 19), "fin": date(2022, 6, 19)},
    {"evento": "Día del Padre", "inicio": date(2023, 6, 18), "fin": date(2023, 6, 18)},
    {"evento": "Día del Padre", "inicio": date(2024, 6, 16), "fin": date(2024, 6, 16)},
    {"evento": "Día del Niño", "inicio": date(2021, 8, 8), "fin": date(2021, 8, 8)},
    {"evento": "Día del Niño", "inicio": date(2022, 8, 7), "fin": date(2022, 8, 7)},
    {"evento": "Día del Niño", "inicio": date(2023, 8, 13), "fin": date(2023, 8, 13)},
    {"evento": "Día del Niño", "inicio": date(2024, 8, 11), "fin": date(2024, 8, 11)},
    {"evento": "Fiestas Patrias", "inicio": date(2021, 9, 18), "fin": date(2021, 9, 19)},
    {"evento": "Fiestas Patrias", "inicio": date(2022, 9, 18), "fin": date(2022, 9, 19)},
    {"evento": "Fiestas Patrias", "inicio": date(2023, 9, 18), "fin": date(2023, 9, 19)},
    {"evento": "Fiestas Patrias", "inicio": date(2024, 9, 18), "fin": date(2024, 9, 19)},
    {"evento": "Cyber Monday", "inicio": date(2021, 10, 4), "fin": date(2021, 10, 6)},
    {"evento": "Cyber Monday", "inicio": date(2022, 10, 3), "fin": date(2022, 10, 5)},
    {"evento": "Cyber Monday", "inicio": date(2023, 10, 2), "fin": date(2023, 10, 4)},
    {"evento": "Cyber Monday", "inicio": date(2024, 9, 30), "fin": date(2024, 10, 2)},
    {"evento": "Black Friday", "inicio": date(2021, 11, 26), "fin": date(2021, 11, 26)},
    {"evento": "Black Friday", "inicio": date(2022, 11, 25), "fin": date(2022, 11, 25)},
    {"evento": "Black Friday", "inicio": date(2023, 11, 24), "fin": date(2023, 11, 24)},
    {"evento": "Black Friday", "inicio": date(2024, 11, 29), "fin": date(2024, 11, 29)},
]

## Creacion de nueva columna de evento, por defecto: "Ninguno"

df['evento'] = 'Ninguno'

## Asignacion de nombre del evento si la fecha cae en los rangos estipulados de lo contrario default

for ev in eventos_skechers:
    inicio = pd.to_datetime(ev['inicio'])
    fin = pd.to_datetime(ev['fin'])
    mask = (df['fecha'] >= inicio) & (df['fecha'] <= fin)
    df.loc[mask, 'evento'] = ev['evento']

## Verificacion rapida de asignaciones:

print("\nEjemplos de fechas con eventos asignados: ")
print(df[df['evento'] != 'Ninguno'][['fecha', 'evento']].head())

# ========================================================================
# PASO 4: FEATURE ENGINEERING - CARACTERIZACION DE VARIABLES DERIVADAS
#           ¿Por qué estas variables?
#            CTR: ayuda a medir el engagement con los anuncios.
#            CPC: te da un control sobre el costo de adquisición.
#            ROAS: es clave para optimizar inversión: cuánto gano por cada peso gastado.
#            es_evento: permite al modelo identificar patrones de comportamiento especiales.
# ========================================================================

## Creacion de variables calculadas

df['ctr_calc'] = df['clics'] / df['impresiones']
df['cpc_calc'] = df['costo'] / df['clics']
df['roas'] = df['valor_conversiones'] / df['costo']

## Reemplazar infinitos (divisiones * 0 ) con NaN y eliminar

df.replace([np.inf, -np.inf], np.nan, inplace= True)
df.dropna(inplace= True)

## Creacion de variable binaria (Valores float):
#           0: No hubo evento
#           1: Si hubo evento

df['es_evento'] = np.where(df['evento'] != 'Ninguno', 1, 0)

## Verificacion de correcta implementacion en DF

print("\nEjemplo de variables generadas:")
print(df[['fecha', 
          'costo', 
          'clics', 
          'valor_conversiones', 
          'ctr_calc', 'cpc_calc', 
          'roas', 
          'es_evento']].head())

# ===============================
# PASO 5: DEFINIR VARIABLES Y DIVIDIR EN TRAIN / TEST
# ===============================

## Definir variable objetivo (ROAS)

objetivo = 'roas'

## Variables predictoras

features = ['costo', 'clics', 'impresiones', 'ctr_calc', 'cpc_calc', 'es_evento']
fecha_corte = pd.to_datetime("2025-02-01")

df_train = df[df['fecha'] < fecha_corte]
df_test = df[df['fecha'] >= fecha_corte]

## X & Y para entrenamiento y prueba

x_train = df_train[features]
y_train = df_train[objetivo]

x_test = df_test[features]
y_test = df_test[objetivo]

print(f"\nEntrenamiento: {x_train.shape[0]} filas")
print(f"Prueba (febrero 2025): {x_test.shape[0]} filas")

# ========================================================================
# PASO 6: ENTRENAMIENTO DE MODELOS Y COMPARACION DE RESULTADOS OBTENIDOS
# ========================================================================

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error , root_mean_squared_error , r2_score

## Funcion de evaluacion para los modelos

def evaluar_modelo(nombre, y_true, y_pred):
    print(f"\n--- {nombre} ---")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"RMSE: {root_mean_squared_error(y_true, y_pred):,.2f}")
    print(f"R2 Score: {r2_score(y_true, y_pred):.4f}")
    
## REGRESION LINEAL

modelo_lr = LinearRegression()
modelo_lr.fit(x_train, y_train)
pred_lr = modelo_lr.predict(x_test)
evaluar_modelo("Regresion lineal", y_test, pred_lr)

## MODELO RANDOM FOREST

modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(x_train, y_train)
pred_rf = modelo_rf.predict(x_test)
evaluar_modelo("Random Forest", y_test, pred_rf)

## MODELO XGBOOST

modelo_xgb = XGBRegressor(n_estimators = 100 , random_state = 42)
modelo_xgb.fit(x_train, y_train)
pred_xgb = modelo_xgb.predict(x_test)
evaluar_modelo("XGBoost", y_test, pred_xgb)

# ===============================
# PASO 7: PREDICCIÓN Y RECOMENDACIÓN PARA FEBRERO 2025
# ===============================

## Generar prediccion para todo el mes de febrero

df_febrero = df_test.copy()
df_febrero['prediccion_rf'] = modelo_rf.predict(x_test)

## Muestra de valores reales vs valores predichos

print("\nPredicciones para febrero 2025:")
print(df_febrero[['fecha', 'costo', 'valor_conversiones', 'prediccion_rf']])

## Calculo / totalizacion

total_real = df_febrero['valor_conversiones'].sum()
total_estimado = df_febrero['prediccion_rf'].sum()
total_inversion = df_febrero['costo'].sum()
roas_estimado = total_estimado / total_inversion if total_inversion > 0 else 0

## RECOMENDACIONES

print("\n--- Resumen y Recomendación de Inversión ---")
print(f"Inversión total en febrero 2025: ${total_inversion:,.2f}")
print(f"Valor real de conversiones: ${total_real:,.2f}")
print(f"Valor estimado por modelo: ${total_estimado:,.2f}")
print(f"ROAS estimado por Random Forest: {roas_estimado:.2f}")

if roas_estimado >= 1:
    print("Recomendación: Mantener o aumentar inversión en Search durante febrero.")
else:
    print("Recomendación: Revisar campañas y optimizar antes de incrementar inversión.")
    
    
## De acuerdo a coeficientes obtenidos se procede a implementar un modelo adicional - SARIMAX (Analisis temporal)

## Para mejorar su precision procederemos a usar multiples regresores exogenos

# ==========================================
# MODELO SARIMAX ROBUSTO CON MULTIVARIABLES
# ==========================================

df['fecha'] = pd.to_datetime(df['fecha'])  # por si acaso
df = df.set_index('fecha')
df = df.sort_index()
exog_cols = ['costo', 'clics', 'impresiones', 'ctr_calc', 'cpc_calc', 'es_evento']

# Entrenamiento (hasta enero 2025)
y_train = df['valor_conversiones']['2021-07-01':'2025-01-31']
X_train = df[exog_cols]['2021-07-01':'2025-01-31']

## Prediccion febrero 2025
X_forecast = df[exog_cols]['2025-02-01':'2025-02-28']

# Modelo SARIMAX manual con estacionalidad semanal
modelo_sarimax = SARIMAX(
    y_train,
    exog=X_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

modelo_fit = modelo_sarimax.fit(disp=False)

# Predecir los próximos N pasos a partir del final de y_train
n_steps = len(X_forecast)
pred_sarimax = modelo_fit.forecast(steps=n_steps, exog=X_forecast)

## Evaluar resultados

inversion_feb = df.loc['2025-02-01':'2025-02-28', 'costo'].sum()
valor_estimado = pred_sarimax.sum()
roas = valor_estimado / inversion_feb if inversion_feb > 0 else 0

print(f"\nTotal estimado de conversiones (SARIMAX): ${valor_estimado:,.2f}")
print(f"Inversión en febrero 2025: ${inversion_feb:,.2f}")
print(f"ROAS estimado (SARIMAX): {roas:.2f}")


# ==========================================
#     VISUALIZACIONES COMPLEMENTARIAS 
# ==========================================

# 1. Comportamiento historico de conversiones e inversion

plt.figure(figsize=(14, 6))
plt.plot(df.index, df['valor_conversiones'], label='Valor de Conversiones', color='green')
plt.plot(df.index, df['costo'], label='Costo de Campañas', color='blue')
plt.title('Evolución de Conversiones y Costo en el tiempo')
plt.xlabel('Fecha')
plt.ylabel('Valor en $')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## ROAS historico por mes 

df['mes_anio'] = df.index.to_period('M')
roas_mensual = df.groupby('mes_anio').apply(lambda x: x['valor_conversiones'].sum() / x['costo'].sum())

roas_mensual.plot(kind='bar', figsize=(14, 6), color='orange')
plt.title('ROAS mensual')
plt.ylabel('ROAS')
plt.xlabel('Mes')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

## Predicciones VS reales para febrero (Modelo SARIMAX)

print(f"Longitud predicción SARIMAX: {len(pred_sarimax)}")

fechas_predichas = pd.date_range(start='2025-02-01', periods=len(pred_sarimax), freq='D')
conversiones_pred = pd.Series(pred_sarimax.values, index=fechas_predichas)
conversiones_real = df['valor_conversiones'].groupby(df.index).sum().reindex(fechas_predichas)

plt.figure(figsize=(14, 6))
plt.plot(conversiones_real.index, conversiones_real, label='Real', marker='o')
plt.plot(conversiones_pred.index, conversiones_pred, label='Predicción SARIMAX', marker='x')
plt.title('Comparación: Conversión Real vs Predicha (Febrero 2025)')
plt.xlabel('Fecha')
plt.ylabel('Valor de Conversiones')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()