
# UN MODELO DE REGRESION SIMPLE: Ancho - Precio

# UN MODELO DE REGRESION MULTIPLE: Peso de motor, consumo ciudad, tamaño de motor - Precio

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
df_autos=df_autos.dropna(subset=['Price', 'width', 'curb-weight','engine-zize','city-mpg'],axis=0)

# Sustiruir ? por '0'
df_autos.loc[df_autos['Price'] == '?', ['Price']] = '0'
df_autos.loc[df_autos['width'] == '?', ['width']] = '0'
df_autos.loc[df_autos['curb-weight'] == '?', ['curb-weight']] = '0'
df_autos.loc[df_autos['engine-zize'] == '?', ['engine-zize']] = '0'
df_autos.loc[df_autos['city-mpg'] == '?', ['city-mpg']] = '0'

# Convertir a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
df_autos["width"]=df_autos["width"].astype("float64")
df_autos["curb-weight"]=df_autos["curb-weight"].astype("float64")
df_autos["engine-zize"]=df_autos["engine-zize"].astype("float64")
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")

# renombrar Variables.
df_autos.rename(columns={"width" : "Ancho"},inplace=True)
df_autos.rename(columns={"Price" : "Precio"},inplace=True)
df_autos.rename(columns={"curb-weight" : "Peso"},inplace=True)
df_autos.rename(columns={"engine-zize" : "Tamanno-motor"},inplace=True)
df_autos.rename(columns={"city-mpg" : "Consumo"},inplace=True)

df2=df_autos[['Precio','Ancho', 'Peso', 'Tamanno-motor', 'Consumo']]

#-------------------------------------------------------------------------------
# Los modelos de regresion son muy sensibles a los outliers, vamos a intentar eliminar aquellos datos que sospechemos que son incorrectos
# Tenemos precios con valores de 0, obviamente no pueden ser y los eliminamos
df2 = df2.drop(df2[df2['Precio']==0].index)
df2 = df2.drop(df2[df2['Peso']==0].index)
df2 = df2.drop(df2[df2['Tamanno-motor']==0].index)
df2 = df2.drop(df2[df2['Consumo']==0].index)




# ==============================================================================
# ==============================================================================
# MODELO DE REGRESION SIMPLE

# La variable dependiente tiene que ser el precio y la independiente el ancho
# Regresión Lineal Simple.
# Seleccionamos las variables: Ancho y precio
x=df2[['Ancho']]
y=df2['Precio']

# Ajustamos las variables al modelo de regresiion lineal
lr.fit(x,y)

# Inferimos valores de y en funcion de x
predichos=lr.predict(x)
# print(predichos)

# Gráfico de Dispersión
sns.regplot(x='Ancho',y='Precio',data=df2,dropna=True,scatter_kws={"color" : "blue"},line_kws={"color" : "red"})
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.title('Dispersión Ancho y Precio Vehículo')
plt.xlabel('Ancho')
plt.ylabel('Precio')
plt.show()

# Grafico Residual
sns.residplot(df2['Ancho'],df2['Precio'],dropna=True,lowess = True)
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.title('Grafico de resiudos')
plt.show()

# Gráfico de Distribución Simple
axl=sns.distplot(df2['Precio'],hist=False,color='r', label='Valor Real')
sns.distplot(predichos,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Modelo simple: Prediccion (a) - Valores reales (r)')
plt.show()



# ==============================================================================
# ==============================================================================
# MODELO DE REGRESION MULTIPLE
Z=df2[['Peso', 'Tamanno-motor', 'Consumo']]

# Ajustamos el modelo
lr.fit(Z,df2['Precio'])
# Predecimos
Pred2=lr.predict(Z)

# Gráfico de Distribución Valores Múltiples.
axl=sns.distplot(df2['Precio'],hist=False,color='r', label='Valor Actual')
sns.distplot(Pred2,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Modelo multiple: Prediccion (a) - Valores reales (r)')
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()