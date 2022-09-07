#-------------------------------------------------------------------------------
# Name:        Ejercicio 12
# Author:      Miguel Ángel
# Created:     28/02/2022
# Copyright:   (c) Miguel Ángel 2022
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


# Utilizando el fichero autos:

    # Crear un modelo polinómico simple

        # Utilizar un 70% de datos para entraniemto
        # Obtener el valor R2 resultante del entrenamiento
        # Polinomio de 3r orden
        # Vdependiente: Precio
        # Vindependiente: Caballos

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#1: CARGAMOS LAS LIBRERIAS QUE NECESITAMOS
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#2: CARGAR LA BASE DE DATOS
Cabeceras=['Symboling','Normalized-Losses','Mark','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','Tamanno-motor','fuel-system','bora','stroke','compression-ratio','Caballos','peak-rpm','Consumo-ciudad','Consumo-carretera','Precio']

# data= pd.read_csv(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\autos.csv", header=None, names = Cabeceras)
data= pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#3: LIMPIAR LOS DATOS

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

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#4A: PARTIR LOS DATOS PARA EL ENTRENAMIENTO
x_train, x_test, y_train, y_test = train_test_split(data['Caballos'], data['Precio'], test_size=0.3)

print("Cantidad de datos en el subconjunto de entrenamiento:", len(x_train))
print("Cantidad de datos en el subconjunto de entrenamiento:", len(x_test))

X_train = np.array(x_train)
Y_train = np.array(y_train)
X_test = np.array(x_test)
Y_test = np.array(y_test)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#4B: ELABORAR EL MODELO
Z = np.polyfit(X_train,Y_train,3)    # z = np.polyfit(vindependiente, vdependiente, grado)
modelo = np.poly1d(Z)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#5: VISULIZAR LOS RESULTADOS DEL MODELO

def PlotPolly(model, vindependiente, vdependiente, Nommbre_grafico, Nombnreindependiente, Nombredependiente):
  print(Nommbre_grafico)
  #genera un array NumPy formado por n números equiespaciados entre dos dados.
  #Su sintaxis es: linspace(valor-inicial, valor-final, número de valores)
  # la funcion np.linspace nos crea una serie de numeros. El primer valor es el minimo del rango, la segunda el máximo y la cantidad de valores que hay en el rango
  # En nuestro codigo, vamos a usar este rango para para modelizar los datos, lo que significa que el minimo sera el minimo de nuestra  prediccion, el maximo el maximo
  #     y la cantidad de valores del rango sera la cantida de tramos que tendra la linea que dibuje nuestro modelo. Es decir, cuantos menos valores, tendrá más aristas
  x_new = np.linspace(min(vindependiente), max(vindependiente), len(vindependiente))

  # Establecemos un Modelo con la relación de X_new y
  # la simbología polinómica p
  y_new = model(x_new)

  plt.plot(vindependiente, vdependiente, '.', x_new, y_new, '-')
  plt.title(Nommbre_grafico)
  ax = plt.gca() # Instancia actual de axes-ax= Area o límites dentro del papel No acepta ni devuelve parámetros
  ax.set_facecolor((0.898, 0.898, 0.898)) #Establecer color en la Cara de los Ejes.
  fig = plt.gcf() # Establecer Figura o Diagrama en 2D
  plt.xlabel(Nombnreindependiente)
  plt.ylabel(Nombredependiente)

  plt.show()
  plt.close()

PlotPolly(modelo, X_train, Y_train, "Valor del precio en funcion de los Caballos", "Caballos", "Precio")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#6: OBTENER EL VALOR DEL R2 TRAS EL ENTRENAMIENTO

