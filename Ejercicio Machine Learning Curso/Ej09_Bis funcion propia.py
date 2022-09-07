#-------------------------------------------------------------------------------
# Name:        Ejercicio 10
# Purpose:
# Author:      Miguel Ángel
# Created:     23/02/2022

#-------------------------------------------------------------------------------
# Primero cargamos las librerias que hacen falta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Abrimos la base de datos con la que vamos a trabajar
data= pd.read_csv( r"drinks.csv")

# Castellanizo y acorto los nombres de las columnas
data = data.rename(columns= {"continent": "continente", "country":"pais", "beer_servings":"cerveza", "spirit_servings":"licor","total_litres_of_pure_alcohol":"puro", "wine_servings":"vino"})

# Puedo hacer una descripcion de los datos para evaluar los datos
print(data.describe())

# -----------------------------------------------------------------------------
# Pasamos al ejercicio, como vamos a trabajar con tres variables, prefiero hacer una funcion que me acorte el trabajo y el código

# PRIMERO: Creamos una funcion que va a trabajar en la creacion del modelo: Polinomialg()
# En la misma funcion he incorporado otra funcion para que grafique los datos  PlotPolly(), por eso necesitamos incorporar muchos datos que no son necesarios para elaborar el modelo

def Polinomialg (grado, vindependiente, vdependiente, Nommbre_grafico, Nombnreindependiente, Nombredependiente):

    z2 = np.polyfit(vindependiente, vdependiente, grado)
    modelo = np.poly1d(z2)
    PlotPolly(modelo, vindependiente, vdependiente, Nommbre_grafico,Nombnreindependiente, Nombredependiente)

        # grado - el grado que queremos ponerle a nuestro polinomio (int)
        # vindependiente - variable independiente (np.array)
        # vdependiente - variable dependiente (np.array)
        # Nombre_grafico - el nombre que le quiero poner al grafico (str)
        # Nombnreindependiente - el nombre de la variable independiente (str)
        # Nombredependiente - el nombre de la variable dependiente (str)


#SEGUNDO: La función para graficar

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
  ax = plt.gca() # Instancia actual de axes-ax= A´rea o límites dentro del papel No acepta ni devuelve parámetros
  ax.set_facecolor((0.898, 0.898, 0.898)) #Establecer color en la Cara de los Ejes.
  fig = plt.gcf() # Establecer Figura o Diagrama en 2D
  plt.xlabel(Nombnreindependiente)
  plt.ylabel(Nombredependiente)

  plt.show()
  plt.close()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Convierto las columnas con las que quiero trabajr en variables con formato np.array
cerveza = np.array(data["cerveza"])
vino = np.array(data["vino"])
licor = np.array(data["licor"])
puro = np.array(data["puro"])

# Grafico mis datos con grado 3
Polinomialg(3, licor, puro, "licor-puro", "licor", "puro")
Polinomialg(3, vino, puro, "vino-puro", "vino", "puro")
Polinomialg(3, cerveza, puro, "cerveza-puro", "cerveza", "puro")

# Conversión Polinomio Múltiple con la Función fit_transform()
y=data['puro']
Z=data[['cerveza','licor','vino']]
pr=PolynomialFeatures(degree=3,include_bias=False)
z_pr=pr.fit_transform(Z)