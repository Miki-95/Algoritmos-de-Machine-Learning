#-------------------------------------------------------------------------------
# Name:        Ejercicio 11 --- Modelo de evaluacón de datps
# Purpose:
# Author:      Miguel Ángel
# Created:     28/02/2022
# Copyright:   (c) Miguel Ángel 2022
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# - 85 % de datos de entrenamiento.
# - Sin generar datos aleatorios.
# - Combinar 4 variables independientes.
# - Conseguir coeficiente R2 mayor al .79.
# - La variable dependiente es el precio

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


# CARGAMOS LOS PAQUETES NECESARIOS
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


# CARGAMOS LOS DATOS CON LOS QUE VAMOS A TRABAJAR
Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','Tamanno-motor','fuel-system','bora','stroke','compression-ratio','Caballos','peak-rpm','Consumo-ciudad','Consumo-carretera','Precio']
# df_autos= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
df_autos= pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

# LIMPIAMOS Y CRAMOS LAS VARIABLES
data = df_autos

# Limpiamos los datos de valores nulos
# Sustiruir ? por '0'
data.loc[data['Precio'] == '?', ['Precio']] = '0'
data.loc[data['Caballos'] == '?', ['Caballos']] = '0'

# Convertir a Float64
data["Precio"]=data["Precio"].astype("float64")
data["Caballos"]=data["Caballos"].astype("float64")

# Elininamos los 0
data =data.drop(data[data['Precio']==0].index)
data =data.drop(data[data['Caballos']==0].index)


# 1: Definir las variables que vamos a emplear
IND = data[['Tamanno-motor','Caballos', 'Consumo-ciudad','Consumo-carretera']]
Y = data['Precio']
# IND.describe()
# IND['Caballos'].describe()   - Si no le quito los valores nulos es un objeto


# 2: Dividir el data set en un subconjunto de entrenamiento y otro de testeo
IND_train, IND_test, Y_train, Y_test = train_test_split(IND,Y, test_size = 0.15)

print("Number of test samples:", IND_test.shape[0])
print("Number of training samples:", IND_train.shape[0])


#3: Seleccionar un modelo para entrenar los datos
# Regresión Lineal Múltiple.
lm=LinearRegression()

lm.fit(IND,data['Precio'])
Yhatm=lm.predict(IND)

width = 10
height = 10
plt.figure(figsize= (width, height))

ax1=sns.distplot(data['Precio'],hist=False,color='r', label='Valor Actual')
sns.distplot(Yhatm, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title("Modelo Lineal Multiple")
plt.xlabel("Variable Compuesta")
plt.ylabel("Precio")
# plt.show()

#4: Probamos como se ajusta el modelo
score = lm.score(IND_test,Y_test)

print("El valor R2 de nuestro modelo es:", score)


# Generar un mensaje que nos diga si nuestro R2 es mayor o igual al valor de 0.79
if score >= 0.79:
    print("Hemos encontrado un modelo con un R2 mayor a 0.79")

else:
    print ("Deberiamos buscar un nuevo modelo que se ajuste mejor a los datos")