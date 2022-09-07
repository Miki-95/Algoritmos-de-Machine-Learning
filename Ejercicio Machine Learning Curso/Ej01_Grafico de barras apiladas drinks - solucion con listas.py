#-------------------------------------------------------------------------------
# Name:        
# Purpose:     Ejercicio 1 pero resuelto con listas
# Author:      Miguel Ángel
# Created:     25/01/2022
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


f = open(r"C:\Users\Jaime\Desktop\Prácticas Data Science MA\Bases de datos ejercicios\drinks.csv", "r")
g = f.readlines()
f.close()

# Hago unas listas para licores por continente. El orden será el que sigue - 0: Europa, 1: Africa, 2: Asia, 3: 0ceania, 4: America

Continentes = ["Europa", "Africa", "Asia", "0ceania", "America"]
Vino = [0,0,0,0,0]
Birra = [0,0,0,0,0]
Licor = [0,0,0,0,0]


#Va a leer ahora las listas, recuerda el orden del fichero drinks - 0: country, 1: beer, 2: spirit, 3: wine, 4: pure alcohol (float), 5: continent

for lista in g:
    #Eliminar el salto de linea
    lista = lista.rstrip()
    #Se convierten las lineas en arreglos con un split
    lista= lista.split(',')

    #Tenemos las lineas


    if lista[5] == 'Asia':
        Vino[2] += int(lista[3])
        Birra[2] += int(lista[1])
        Licor[2] += int(lista[2])

    if lista[5] == 'Africa':
        Vino[1] += int(lista[3])
        Birra[1] += int(lista[1])
        Licor[1] += int(lista[2])

    if lista[5] == 'Europe':
        Vino[0] += int(lista[3])
        Birra[0] += int(lista[1])
        Licor[0] += int(lista[2])

    if lista[5] == 'Oceania':
        Vino[3] += int(lista[3])
        Birra[3] += int(lista[1])
        Licor[3] += int(lista[2])

    if lista[5] == 'North America':
        Vino[4] += int(lista[3])
        Birra[4] += int(lista[1])
        Licor[4] += int(lista[2])

    if lista[5] == 'South America':
        Vino[4] += int(lista[3])
        Birra[4] += int(lista[1])
        Licor[4] += int(lista[2])



listaCV=[]
for i in range(len(Birra)):
   j= Birra[i]+Vino[i]
   listaCV.append(j)


indice = np.arange(len(Continentes))

## Se crean las primeras barras
plt.bar(indice, Vino, label='Vino')

## Se crean las segundas barras y se apilan sobre las primeras
plt.bar(indice, Birra, label='Cerveza',  bottom=Vino)

## Se crean las terceras barras y se apilan sobre las segundas
plt.bar(indice, Licor, label='Licor',  bottom=listaCV)

plt.xticks(indice, Continentes)
plt.ylabel("Bebidas alcohólicas")
plt.xlabel("Continentes")
plt.title('Consumo de bebidas alcohólicas por continente')
plt.legend()

plt.show()

