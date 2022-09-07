#-------------------------------------------------------------------------------
# Name:        Ejercicio 14

# Purpose:      1 - Hacer un modelo de regresion lineal y un modelo de polinómico
#               2 - Graficar ambos modelos con respecto a los valores reales y compararlos
# Author:      Miguel Ángel
# Created:     23/03/2022
#-------------------------------------------------------------------------------

# 1: Cargamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
lr = LinearRegression ()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# 2:  Abrimos el archivo CSV con todos los datos que  necesitamos
China = pd.read_csv(r"PibChina.csv")

# 3: Graficar los datos
# Visualizar las variables
X = China[['Year']]
Y = China['Value']

def Visualizar_datos ():
    plt.scatter(X, Y)
    plt.title("Evolución del PIB de China")
    plt.show()
Visualizar_datos ()

# 4 Modelo de regresion lineal simple
lr.fit(X,Y)
Predichos_lineal = lr.predict(X)
R2s=lr.score(X,Y)

# GRAFICOS DEl MODELO LINEAL SIMPLE   -    Estos son los que he hecho yo, se pueden borrar
def Grafico_simple():
    # Gráfico de Dispersión
    plt.subplot2grid((2,2), (0,0),rowspan=1)
    sns.regplot(x='Year',y='Value',data=China,dropna=True,scatter_kws={"color" : "blue"},line_kws={"color" : "red"})
    plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
    plt.title("Modelo - Datos")

    # Grafico Residual
    plt.subplot2grid((2,2), (0,1),rowspan=1)
    sns.residplot(China['Year'],China['Value'],dropna=True,lowess = True)
    plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
    plt.title("Residuos")

    # Gráfico de Distribución Simple
    plt.subplot2grid((2,2), (1,0),colspan=2)
    axl=sns.distplot(China['Value'],hist=False,color='r', label='Valor Real')
    sns.distplot(Predichos_lineal,hist=False,color='b', label='Valores Ajustados',ax=axl)
    plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
    plt.title("Valores reales - predichos")

    plt.show()

Grafico_simple()
#-------------------------------------------------------------------------------

# 5- MODELO POLINÓMICO
pr = PolynomialFeatures(degree=2, include_bias=False )
Z = China[['Year'[:]]]
Z = pr.fit_transform(Z)

# Creamos una tupla con el nombre del modelo y su constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# Ponemos la tupla como un argumento en el constructor Pipeline
Pipe = Pipeline(Input)

# Transformation, Fitting y prediccion
Pipe.fit(Z,Y)
Prediccion_Polinomico=Pipe.predict(Z)
R2=Pipe.score(Z,Y)
print("El valor de la R2 del modelo polinómico es es:", R2)

#-------------------------------------------------------------------------------
# GRAFICOS DEl MODELO LINEAL SIMPLE
def Grafico_multiple():

    # Gráfico de Distribución Valores Polinómicos Múltiples.
    axl=sns.distplot(China['Value'],hist=False,color='r', label='Valor Real')
    sns.distplot(Prediccion_Polinomico,hist=False,color='b', label='Valores Ajustados',ax=axl)
    plt.ylabel ("PIB")
    plt.xlabel ("Modelo Polinomico ")
    plt.title("Valores Predichos - Valores Reales")
    plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
    plt.show()

    # Gráfico de Dispersion Valores Polinómicos Múltiples - Precio.
    plt.scatter(Prediccion_Polinomico, Y, color='g')
    plt.ylabel ("PIB")
    plt.xlabel ("Modelo Polinomico ")
    plt.title("PIB - Modelo Polinomico ")
    plt.show()
Grafico_multiple()

#-------------------------------------------------------------------------------

# 5- MODELO POLINÓMICO 3
pr = PolynomialFeatures(degree=3, include_bias=False )
Z = China[['Year'[:]]]
Z = pr.fit_transform(Z)

# Creamos una tupla con el nombre del modelo y su constructor
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# Ponemos la tupla como un argumento en el constructor Pipeline
Pipe = Pipeline(Input)

# Transformation, Fitting y prediccion
Pipe.fit(Z,Y)
Prediccion_Polinomico=Pipe.predict(Z)
R2=Pipe.score(Z,Y)
print("El valor de la R2 del modelo polinómico 3 es es:", R2)


# GRAFICOS DEl MODELO LINEAL SIMPLE
def Grafico_multiple3():

    # Gráfico de Distribución Valores Polinómicos Múltiples.
    axl=sns.distplot(China['Value'],hist=False,color='r', label='Valor Real')
    sns.distplot(Prediccion_Polinomico,hist=False,color='b', label='Valores Ajustados',ax=axl)
    plt.ylabel ("PIB")
    plt.xlabel ("Modelo Polinomico  3")
    plt.title("Valores Predichos - Valores Reales")
    plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
    plt.show()

    # Gráfico de Dispersion Valores Polinómicos Múltiples - Precio.
    plt.scatter(Prediccion_Polinomico, Y, color='g')
    plt.ylabel ("PIB")
    plt.xlabel ("Modelo Polinomico  3")
    plt.title("PIB - Modelo Polinomico ")
    plt.show()
# Grafico_multiple3()