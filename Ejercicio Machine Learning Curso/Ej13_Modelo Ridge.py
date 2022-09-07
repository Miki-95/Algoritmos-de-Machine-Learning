#-------------------------------------------------------------------------------
# Name:        Ejercicio 13
# Purpose:     Comparar alfas y hacer R2
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

##############################################################
# Carga y limpieza de los datos
##############################################################

# Abrimos el archivo CSV con todos los datos que  necesitamos
Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
# df_autos= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

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

##############################################################
# Modelamos
##############################################################
# Preparamos las variables con las que vamos a trabajar:
Y =df_autos['Price']
X =df_autos[['horsepower','highway-mpg','engine-zize']]

# Dividimos los datos
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

# Transformacion de los datos a polinomio de grado 3
pr = PolynomialFeatures(degree=3, include_bias=False )
x_test_pr = pr.fit_transform(x_test)
x_train_pr = pr.fit_transform(x_train)

# Creamos un objeto de ridgeRegresion
RidgeModel=Ridge(alpha=0.1)

# Ajustamos al modelo
RidgeModel.fit(x_train_pr, y_train)

# Hacemos predicciones con el modelo
Predichos = RidgeModel.predict(x_test_pr)

# Buscamos el valor de alfa que minimiza el error
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,10))

for alfa in ALFA:
    RidgeModel = Ridge(alpha=alfa)
    RidgeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RidgeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RidgeModel.score(x_train_pr, y_train))

# Visualizamos los valores de R2 para las distintas alfa
plt.figure(figsize=(12, 10))
plt.plot(ALFA,Rsqu_test, label='Datos de Validacion')
plt.plot(ALFA,Rsqu_train, 'r', label='Datos Entrenamiento ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()
plt.close()
