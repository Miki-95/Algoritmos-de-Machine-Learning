#-------------------------------------------------------------------------------
# Name:        Ejercicio 19 - Fase 02
# Purpose:      Modelo polinomico con la base cars empleando la tecnica de pipelines
#
# Author:      Miguel Ángel
#
# Created:     15/03/2022

#-------------------------------------------------------------------------------

# 1: Cargamos las librerias


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures




# 2:  Abrimos el archivo CSV con todos los datos que  necesitamos

Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
# df_autos= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)


# Limpiamos los datos, muchos de ellos vienen con valores que estan vacios y por tanto nos guarda como un objeto molesto

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

# Tenemos precios con valores de 0, obviamente no pueden ser y los eliminamos
df2 = df_autos
df2 = df2.drop(df2[df2['Price']==0].index)
df2 = df2.drop(df2[df2['curb-weight']==0].index)
df2 = df2.drop(df2[df2['engine-zize']==0].index)
df2 = df2.drop(df2[df2['city-mpg']==0].index)
df2 = df2.drop(df2[df2['horsepower']==0].index)


# 3: Visualizamos las variables que queremos trabajar


# Visualizar las variables

def graficar_variables ():
    X1 = df2['curb-weight']
    X2 = df2['engine-zize']
    X3 = df2['city-mpg']
    X4 = df2['horsepower']
    Y = df2['Price']

    plt.subplot2grid((2,2), (0,0),rowspan=1)
    plt.scatter(X1, Y, color='r')
    plt.ylabel ("precio")
    plt.title("Peso_vacio")

    plt.subplot2grid((2,2), (0,1),rowspan=1)
    plt.scatter(X2, Y, color='g')
    plt.ylabel ("precio")
    plt.title("Tamanno_motor")

    plt.subplot2grid((2,2), (1,0),colspan=1)
    plt.scatter(X3, Y, color='y')
    plt.ylabel ("precio")
    plt.title("Consumo_carretera")

    plt.subplot2grid((2,2), (1,1),colspan=1)
    plt.scatter(X4, Y)
    plt.ylabel ("precio")
    plt.title("Caballos")

    plt.show()

graficar_variables()

# Definir las variables:

Z = df2 [['horsepower', 'curb-weight', 'engine-zize', 'highway-mpg']]
y =df2 ["Price"]

pr = PolynomialFeatures(degree=3, include_bias=False )
Z = pr.fit_transform(Z)

# Creamos una tupla con el nombre del modelo y su constructor

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# Ponemos la tupla como un argumento en el constructor Pipeline

Pipe = Pipeline(Input)

# Transformation, Fitting y prediccion


Pipe.fit(Z,y)
Yhat=Pipe.predict(Z)
R2=Pipe.score(Z,y)
print("El valor de la R2 es:", R2)


# Gráfico de Distribución Valores Polinómicos Múltiples.
axl=sns.distplot(df2['Price'],hist=False,color='r', label='Valor Real')
sns.distplot(Yhat,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.ylabel ("Precio")
plt.xlabel ("Modelo Polinomico Multiple")
plt.title("Valores Predichos - Valores Reales")
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()


# Gráfico de Dispersion Valores Polinómicos Múltiples - Precio.
plt.scatter(Yhat, y, color='g')
plt.ylabel ("Precio")
plt.xlabel ("Modelo Polinomico Multiple")
plt.title("Precio - Modelo Polinomico Multiple")
plt.show()