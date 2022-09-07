#-------------------------------------------------------------------------------
# Name:        Ejercicio 19 - Fase 01
# Purpose:     Crear modelos con el fichero drinks
#
# Author:      Miguel Ángel
#
# Created:     10/03/2022
# Copyright:   (c) Jaime 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------


# 1: Importar las librerias:


import pandas as pd
import numpy as np

import cx_Oracle as cx

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

#-------------------------------------------------------------------------------
# 2: Creamos conexiones con Oracle

try:
    dsn_tns = cx.makedsn('localhost', '1521', 'XE')
    con = cx.connect(user='Curso_DataScience', password='Flebotomo95', dsn=dsn_tns)
    c = con.cursor()

except:
    print("Error en la conexion con Oracle")

#-------------------------------------------------------------------------------
# 3: Cargar los datos en la base de datos


c.execute("delete from drinkb where registro>0")

f = open(r"drinks.csv", "r")
# f = open(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\drinks.csv", "r")
g = f.readlines()
f.close()

key = 1

try:
    for linea in g:
            # Remover salto de línea
            linea = linea.rstrip()
            # Ahora convertimos la línea a arreglo con split
            separador = ","
            lista = linea.split(",")
            # Tenemos la lista.
            if lista[0]!= 'country':
                c.execute ("insert into drinkB(Registro, Pais, Cerveza, Licor, Vino, Puro, Continente) values ( :1,:2,:3,:4,:5,:6,:7)", (key, lista[0],int(lista[1]),int(lista[2]),int(lista[3]),float(lista[4]),lista[5]))
                key = key+1
    print("Se han registrado los datos")
except:
    print("Error rellenando la tabla")


#-------------------------------------------------------------------------------
# CARGAR LOS DATOS AQUI


c.execute('select * from drinkB')

Registro = []
pais = []
cerveza = []
licor = []
vino = []
puro = []
continente = []
Predicho = []

for row in c:
    Registro.append(row[0])
    pais.append(row[1])
    cerveza.append(row[2])
    licor.append(row[3])
    vino.append(row[4])
    puro.append(row[5])
    continente.append(row[6])
    Predicho.append(0)

data = {"Registro": Registro, "Pais": pais, "Cerveza":cerveza, "Licor":licor, "Vino":vino, "Puro":puro, "Continente":continente, "Predicho":Predicho}
data = pd.DataFrame(data)

data = data.replace({"North America": "America", "South America": "America"})

print(data)
#-------------------------------------------------------------------------------
# Visualizar las variables

X1 = data['Vino']
X2 = data['Cerveza']
X3 = data['Licor']
Y = data['Puro']

plt.subplot2grid((2,2), (0,0),rowspan=1)
plt.scatter(X1, Y)
plt.title("Vino")

plt.subplot2grid((2,2), (0,1),rowspan=1)
plt.scatter(X2, Y)
plt.title("Cerveza")

plt.subplot2grid((2,2), (1,0),colspan=2)
plt.scatter(X3, Y)
plt.title("Licor")

plt.show()

#-------------------------------------------------------------------------------
# ENTRAMOS EN LA GENERACION DE MODELOS

# 1: Definir las variables que vamos a emplear

Z = data[['Vino', 'Cerveza', 'Licor']]
y = data['Puro']


lr = LinearRegression()

# Regresión Lineal Múltiple.
lm=LinearRegression()
lm.fit(Z,y)
Yhatm=lm.predict(Z)

# Calcular la R2

R2 = lm.score(Z,y)
print("El valor de R2 es: ", R2)

# Calcular la R2 ajustada

def R2_ajustado (R2, N, P):
    # R2 sin ajustar
    # N cantidad de elementos
    # P: numero de variables predictoras

    R2A    = 1-(1-R2) * (N-1)/(N-P-1)
    print("El valor de R2 ajustado es: ", R2A)
    return R2A

R2_ajustado (R2, len(Z), 3)

#-------------------------------------------------------------------------------
# GRAFICAMOS LOS RESULTADOS


width = 10
height = 10
plt.figure(figsize= (width, height))

ax1=sns.distplot(y,hist=False,color='r', label='Valor Real')
sns.distplot(Yhatm, hist=False, color="b", label="Valor ajustado" , ax=ax1)
plt.title("Modelo Lineal Multiple valores reales y predichos")
plt.xlabel("Z: Cerveza + Vino + Licor")
plt.ylabel("Puro")
plt.show()

# Diagrama de caja y bigotes
bp = data.boxplot(column = ['Puro'], by= "Continente")
plt.title("Modelo Lineal Multiple valores reales")
plt.xlabel("Z: Cerveza + Vino + Licor")
plt.ylabel("Puro")
plt.show()


# ----------------------------------------------------------------------
# CERRAR LA BASE DE DATOS


con.commit()
c.close()
con.close()
