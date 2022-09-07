#-------------------------------------------------------------------------------
# Name:        Ejercicio 10
# Purpose:
# Author:      Miguel Ángel
# Created:     24/02/2022
# Copyright:   (c) Miguel Ángel 2022
#-------------------------------------------------------------------------------

#1: Cargamos las librerias que nos hacen falta
import pandas as pd # Manipulacion de datos y tablas
import numpy as np # Manipulacion de datos y tablas

import matplotlib.pyplot as plt   # Algunas graficaciones
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures  # Regresion polinomica
from sklearn.pipeline import Pipeline                # Para hacer pipelines
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#2: Cargamos la base de datos con la que vamos a trabajar
data = pd.read_csv(r"drinks.csv")

# Renombramos las columnas
data.rename(columns={"continent":"Continente", "spirit_servings":"Licor", "beer_servings":"Cerveza", "total_litres_of_pure_alcohol":"Puro", "wine_servings":"Vino", "country":"Pais"}, inplace=True)

print(data.head())
print(data.describe())

#3: Pasamos al ejercicio en sí
# Hay que hacer una regresion multiple polinomial
# Creo la variable multiple con las tres variables independientes que deseamos: vino, cerveza y licor
# Creamos las variable dependiente: Puro
Z = data[['Vino', 'Cerveza', 'Licor']]
y = data['Puro']

# Me voy para arriba a cargar PolynomialFeatures
# from sklearn.preprocessing import PolynomialFeatures
# acorto la funcion PolynomialFeatures con un grado 2

# Creamos el input para el pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe

# Modelizamos
pr=PolynomialFeatures(degree=2,include_bias=False)
z_pr=pr.fit_transform(Z)

#Objeto Pipeline
pipe=Pipeline(Input)
pipe.fit(z_pr,y)
Yhat=pipe.predict(z_pr)
R2=pipe.score(z_pr,y)

# Gráfico de Distribución Valores Polinómicos Múltiples.
axl=sns.distplot(data['Puro'],hist=False,color='r', label='Valor Actual')
sns.distplot(Yhat,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.title("Ajuste vs datos reales del alcohol puro")
plt.xlabel("Alcohol puro")
plt.ylabel("Proporcion de alcohol puro")
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()
plt.close()
