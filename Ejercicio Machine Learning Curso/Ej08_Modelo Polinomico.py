#-------------------------------------------------------------------------------
# Name:        Ejercicio Nº 8. Regresión lineal polinómico Simple.
# Author:      Miguel Ángel Gómez Molinero.
# Created:     22/02/2022
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
from seaborn import load_dataset
import seaborn as sns
#from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# ==============================================================================

#### Abrir base de datos
# Cabeceras
Cabeceras=['Symboling','Normalized-Losses','Make','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
#Cargamos los datos
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)

# Filtro estructura de valores y dejo aquellos no nulos.
df_autos=df_autos.dropna(subset=['Price', 'width', 'curb-weight','engine-zize','city-mpg','horsepower'],axis=0)

# Sustiruir ? por '0'
df_autos.loc[df_autos['Price'] == '?', ['Price']] = '0'
df_autos.loc[df_autos['city-mpg'] == '?', ['city-mpg']] = '0'
df_autos.loc[df_autos['horsepower'] == '?', ['horsepower']] = '0'

# Convertir a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")
df_autos["horsepower"]=df_autos["horsepower"].astype("float64")

# renombrar Variables
df_autos.rename(columns={"Price" : "Precio"},inplace=True)
df_autos.rename(columns={"city-mpg" : "Consumo_Ciudad"},inplace=True)
df_autos.rename(columns={"horsepower" : "Caballos"},inplace=True)

df2=df_autos[['Precio','Consumo_Ciudad', 'Caballos']]

# Tenemos valores de 0, los eliminamos
df2 = df2.drop(df2[df2['Precio']==0].index)
df2 = df2.drop(df2[df2['Consumo_Ciudad']==0].index)
df2 = df2.drop(df2[df2['Caballos']==0].index)

# ==============================================================================
# ==============================================================================

# PREPARAMOS LAS VARIABLES DE NUESTRO MODELO POLINOMICO Caballos - Precio

X = np.array(df2['Caballos'])
Y = np.array (df2['Precio'])

plt.scatter(X,Y)
plt.xlabel("Caballos")
plt.ylabel("Precio")

Z = np.polyfit(X, Y, 3)
P = np.poly1d(Z)
print ("Coeficientes:",Z)
print("Modelo  :",P)

plt.show()
# ==============================================================================
# ==============================================================================

# PREPARAMOS LAS VARIABLES DE NUESTRO MODELO POLINOMICO Consumo - Precio
x = np.array(df2['Consumo_Ciudad'])

plt.scatter(x,Y)
plt.xlabel("Consumo_Ciudad")
plt.ylabel("Precio")

z = np.polyfit(x, Y, 3)
p = np.poly1d(z)
print ("Coeficientes:",z)
print("Modelo  :",p)

plt.show()

# ==============================================================================
# ==============================================================================

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    
    minimo = int(round(min(independent_variable)))
    maximo = int(round(max(independent_variable)))
    longitud = len(independent_variable)

    x_new = np.linspace(minimo, maximo, longitud)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polinomio Segun Modelo')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Precio Coche')

    plt.show()
    plt.close()

PlotPolly(P, X, Y, 'Caballos')
PlotPolly(p, x, Y, 'Consumo_Ciudad')

# ==============================================================================
# ==============================================================================
# PREPARAMOS LAS VARIABLES DE NUESTRO MODELO POLINOMICO Consumo - Precio

print ("Coeficientes del modelo 1:",Z)
print("Modelo 1 :",P)

print ("Coeficientes del modelo 2:",z)
print("Modelo 2 :",p)