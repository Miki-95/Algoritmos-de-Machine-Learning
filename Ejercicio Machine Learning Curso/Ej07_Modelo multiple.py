# UN MODELO DE REGRESION MULTIPLE: Precio - Libre


#-------------------------------------------------------------------------------
# CARGAMOS LAS LIBRERIAS QUE NOS HACEN FALTA

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y análisis
# ==============================================================================
import statsmodels.api as sm
from scipy import stats

# Modelado
# ==============================================================================

from sklearn.linear_model import LinearRegression
lr = LinearRegression ()

# ==============================================================================
# ==============================================================================
#### Abrir base de datos
# Cabeceras
Cabeceras=['Symboling','Normalized-Losses','Make','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
#Cargamos los datos
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

#-------------------------------------------------------------------------------
# Filtro estructura de valores y dejo aquellos no nulos.
df_autos=df_autos.dropna(subset=['Price', 'width', 'curb-weight','engine-zize','city-mpg','horsepower'],axis=0)

# Sustiruir ? por '0'
df_autos.loc[df_autos['Price'] == '?', ['Price']] = '0'
df_autos.loc[df_autos['width'] == '?', ['width']] = '0'
df_autos.loc[df_autos['curb-weight'] == '?', ['curb-weight']] = '0'
df_autos.loc[df_autos['engine-zize'] == '?', ['engine-zize']] = '0'
df_autos.loc[df_autos['city-mpg'] == '?', ['city-mpg']] = '0'
df_autos.loc[df_autos['horsepower'] == '?', ['horsepower']] = '0'

# Convertir a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
df_autos["width"]=df_autos["width"].astype("float64")
df_autos["curb-weight"]=df_autos["curb-weight"].astype("float64")
df_autos["engine-zize"]=df_autos["engine-zize"].astype("float64")
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")
df_autos["horsepower"]=df_autos["horsepower"].astype("float64")

# renombrar Variables.
df_autos.rename(columns={"width" : "Ancho"},inplace=True)
df_autos.rename(columns={"Price" : "Precio"},inplace=True)
df_autos.rename(columns={"curb-weight" : "Peso"},inplace=True)
df_autos.rename(columns={"engine-zize" : "Tamanno-motor"},inplace=True)
df_autos.rename(columns={"city-mpg" : "Consumo"},inplace=True)
df_autos.rename(columns={"horsepower" : "Caballos"},inplace=True)

df2=df_autos[['Precio','Ancho', 'Peso', 'Tamanno-motor', 'Consumo', 'Caballos']]

#-------------------------------------------------------------------------------
# Los modelos de regresion son muy sensibles a los outliers, vamos a intentar eliminar aquellos datos que sospechemos que son incorrectos

# Tenemos valores de 0, los eliminamos
df2 = df2.drop(df2[df2['Precio']==0].index)
df2 = df2.drop(df2[df2['Peso']==0].index)
df2 = df2.drop(df2[df2['Tamanno-motor']==0].index)
df2 = df2.drop(df2[df2['Consumo']==0].index)
df2 = df2.drop(df2[df2['Caballos']==0].index)


# ==============================================================================
# ==============================================================================

# MODELO DE REGRESION MULTIPLE: Precio - Libre

y=df2['Precio']
Z=df2[['Caballos', 'Tamanno-motor', 'Consumo']]
lr.fit(Z,df2['Precio'])
predichos=lr.predict(Z)

# Gráfico de Distribución Valores Múltiples.
axl=sns.distplot(df2['Precio'],hist=False,color='r', label='Valor Actual')
sns.distplot(predichos,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Modelo multiple: Prediccion (a) - Valores reales (r)')
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()