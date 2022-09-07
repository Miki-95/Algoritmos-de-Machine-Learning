#-------------------------------------------------------------------------------
# Name:        Ejercicio 19 - Fase 3
# Purpose:     Usar el modelo de arista
#
# Author:      Miguel Ángel
#
# Created:     15/03/2022
# Copyright:   (c) Miguel Ángel 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

# 1: Cargamos las librerias


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# 2:  Abrimos el archivo CSV con todos los datos que  necesitamos

Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
# df_autos= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)
# df_autos = pd.read_csv(r"C:\Users\Miguel Ángel\Desktop\DATA SCIENCE\Curso - Cesur Formacion\autos.csv", header=None, names = Cabeceras)



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
df_autos.loc[df_autos['highway-mpg'] == '?', ['highway-mpg']] = '0'

# Convertir a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
df_autos["width"]=df_autos["width"].astype("float64")
df_autos["curb-weight"]=df_autos["curb-weight"].astype("float64")
df_autos["engine-zize"]=df_autos["engine-zize"].astype("float64")
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")
df_autos["horsepower"]=df_autos["horsepower"].astype("float64")
df_autos.loc[df_autos['highway-mpg'] == '?', ['highway-mpg']] = '0'

# Tenemos precios con valores de 0, obviamente no pueden ser y los eliminamos
df2 = df_autos
df2 = df2.drop(df2[df2['Price']==0].index)
df2 = df2.drop(df2[df2['curb-weight']==0].index)
df2 = df2.drop(df2[df2['engine-zize']==0].index)
df2 = df2.drop(df2[df2['city-mpg']==0].index)
df2 = df2.drop(df2[df2['horsepower']==0].index)
df_autos.loc[df_autos['highway-mpg'] == '?', ['highway-mpg']] = '0'



# Crear las varibales
y=df_autos['Price'].values
Z=df_autos[['Fuel-Type','num-of-doors','horsepower','engine-zize','highway-mpg']].values


# Convertir Valores varible categorica en Valores numérico.
#Columna Fuel-Type
Fuel01 = preprocessing.LabelEncoder()
Fuel01.fit(['gas','diesel'])
Z[:,0] = Fuel01.transform(Z[:,0])
#print(XX[:,0])

# Columna num-of-doors
Npuertas01=preprocessing.LabelEncoder()
Npuertas01.fit(['two','four','?'])
Z[:,1]=Npuertas01.transform(Z[:,1])
#print(XX[:,1])




pr=PolynomialFeatures(degree=3,include_bias=False)
z_pr=pr.fit_transform(Z)
# pipelines
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

#Objeto Pipeline
pipe=Pipeline(Input)
pipe.fit(z_pr,y)
YhatP=pipe.predict(z_pr)
R2f2=pipe.score(z_pr,y)




# Diseñar Gráficos Distribución
axl=sns.distplot(df_autos['Price'],hist=False,color='r', label='Valor Actual')
sns.distplot(YhatP,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.xlabel("Comparación Precios")
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()

# Diseñar Gráficos de Dispersión
sns.regplot(x = y,y =YhatP,
        dropna = True,scatter_kws={"color": "black"}, line_kws={"color": "red"})
plt.show()



# Tercera Fase Evalución y Refinamiento con Apoyandose el Modelo Arista.
# Fichero que vamos a utilizar es el autos.cvs.
pr=PolynomialFeatures(degree=7)
x_train,x_test,y_train,y_test=train_test_split(Z,y,test_size=0.3,random_state=0)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
Model_arista=Ridge(alpha=0.75)
Model_arista.fit(x_train_pr, y_train)
yhat = Model_arista.predict(x_train_pr)
print(Model_arista.score(x_train_pr,y_train))

# Diseñar Gráficos Distribución
axl=sns.distplot(df_autos['Price'],hist=False,color='r', label='Valor Actual')
sns.distplot(yhat,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.xlabel("Comparación Precios")
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()











