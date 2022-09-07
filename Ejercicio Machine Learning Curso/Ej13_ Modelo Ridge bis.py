#-------------------------------------------------------------------------------
# Name:        Ejercicio 13
# Purpose:
#               - Modelo de arista con rejilla
#               - 5 variables  -  Precio
#               - R2
# Author:      Miguel Ángel
# Created:     22/03/2022
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
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# 2:  Abrimos el archivo CSV con todos los datos que  necesitamos
Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
# df_autos= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

# Sustiruir ? por '0'
df_autos.loc[df_autos['Price'] == '?', ['Price']] = '0'
df_autos.loc[df_autos['width'] == '?', ['width']] = '0'
df_autos.loc[df_autos['curb-weight'] == '?', ['curb-weight']] = '0'
df_autos.loc[df_autos['engine-zize'] == '?', ['engine-zize']] = '0'
df_autos.loc[df_autos['city-mpg'] == '?', ['city-mpg']] = '0'
df_autos.loc[df_autos['highway-mpg'] == '?', ['highway-mpg']] = '0'
df_autos.loc[df_autos['horsepower'] == '?', ['horsepower']] = '0'

# Convertir a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
df_autos["width"]=df_autos["width"].astype("float64")
df_autos["curb-weight"]=df_autos["curb-weight"].astype("float64")
df_autos["engine-zize"]=df_autos["engine-zize"].astype("float64")
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")
df_autos["highway-mpg"]=df_autos["highway-mpg"].astype("float64")
df_autos["horsepower"]=df_autos["horsepower"].astype("float64")

# Tenemos valores de 0 que en realidad son NA, los eliminamos
df2 = df_autos
df2 = df2.drop(df2[df2['Price']==0].index)
df2 = df2.drop(df2[df2['curb-weight']==0].index)
df2 = df2.drop(df2[df2['engine-zize']==0].index)
df2 = df2.drop(df2[df2['city-mpg']==0].index)
df2 = df2.drop(df2[df2['highway-mpg']==0].index)
df2 = df2.drop(df2[df2['horsepower']==0].index)

# Preparamos las variables con las que vamos a trabajar:
Y =df_autos['Price']
X =df_autos[['horsepower','highway-mpg','engine-zize','curb-weight','city-mpg']]

# Dividimos los datos
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

# Rejillas de Datos Normalizados
# Cremos un diccionario con los valores de la rejilla
parametros= [{'alpha': [1,10, 100, 1000],'normalize': [True,False]}]

# Creamos un objeto de ridge regions:
RR=Ridge()
RR
# Creamos un objeto que grid search
Grid = GridSearchCV(RR, parametros,cv=4,return_train_score=True)
# cv: Fracciona los datos en tantos grupos como le pidamos para valorar el modleo

# Ajustamos el modelo
Grid.fit(X, Y)

# Guardamos el estinmador con mejores parametros
MejorRR=Grid.best_estimator_

# Probamos nuestro modelo con los datos de prueba
print("El mejor R2 es:")
scores=Grid.cv_results_

print("Los R2 para los datos de testeo son:")
print(scores['mean_test_score'])
print("Los R2 para los datos de entrenamiento son:")
print(scores['mean_train_score'])
print("El mejor R2 para los datos de testeo es:")
print(MejorRR.score(x_test, y_test))
print("El mejor R2 para los datos de entrenamiento es:")
print(MejorRR.score(x_train, y_train))
