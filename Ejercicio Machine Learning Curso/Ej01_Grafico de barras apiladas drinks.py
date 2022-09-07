#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:     Gráfico de barras apiladas
# Author:      Miguel Ángel
# Created:     24/01/2022
# Copyright:   (c) Miguel Ángel 2022
#-------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt


f = open("drinks.csv", "r")
g = f.readlines()
f.close()


#Creamos los contadores de los continentes
#Vino
WAsia = 0
WAfrica = 0
WAmerica = 0
WOceania = 0
WEuropa = 0
#Cerveza
BAsia = 0
BAfrica = 0
BAmerica = 0
BOceania = 0
BEuropa = 0
#Licor
SAsia = 0
SAfrica = 0
SAmerica = 0
SOceania = 0
SEuropa = 0

#Leer y filtrar por continente

for lista in g:
    #Eliminar el salto de linea
    lista = lista.rstrip()
    #Se convierten las lineas en arreglos con un split
    lista= lista.split(',')

    #Tenemos las lineas
    if lista[5] == 'Asia':
        WAsia = WAsia + int(lista[3])
        BAsia = BAsia + int(lista[1])
        SAsia = SAsia + int(lista[2])

    if lista[5] == 'Africa':
        WAfrica = WAfrica + int(lista[3])
        BAfrica = BAfrica + int(lista[1])
        SAfrica = SAfrica + int(lista[2])

    if lista[5] == 'Europe':
        WEuropa= WEuropa + int(lista[3])
        BEuropa = BEuropa + int(lista[1])
        SEuropa = SEuropa + int(lista[2])

    if lista[5] == 'Oceania':
        WOceania = WOceania + int(lista[3])
        BOceania = BOceania+ int(lista[1])
        SOceania = SOceania + int(lista[2])

    if lista[5] == 'North America':
        WAmerica = WAmerica + int(lista[3])
        BAmerica = BAmerica + int(lista[1])
        SAmerica = SAmerica + int(lista[2])

    if lista[5] == 'South America':
        WAmerica = WAmerica + int(lista[3])
        BAmerica = BAmerica + int(lista[1])
        SAmerica = SAmerica + int(lista[2])



grupos = ['Europa', 'America', 'Oceania', 'Asia', 'Africa' ]
#grupos = ['Grupo 1', 'Grupo 2', 'Grupo 3', 'Grupo 4']
Vino = [WEuropa,WAmerica, WOceania, WAsia, WAfrica]
Licor = [SEuropa,SAmerica, SOceania, SAsia, SAfrica]
Cerveza = [BEuropa,BAmerica, BOceania, BAsia, BAfrica]



listaCV=[]
for i in range(len(Cerveza)):
    j= Cerveza[i]+Vino[i]
    listaCV.append(j)

indice = np.arange(len(grupos))

## Se crean las primeras barras
plt.bar(indice, Vino, label='Vino')

## Se crean las segundas barras y se apilan sobre las primeras
plt.bar(indice, Cerveza, label='Cerveza',  bottom=Vino)

## Se crean las terceras barras y se apilan sobre las segundas
plt.bar(indice, Licor, label='Licor',  bottom=listaCV)

plt.xticks(indice, grupos)
plt.ylabel("Bebidas alcohólicas")
plt.xlabel("Grupos")
plt.title('Consumo de bebidas alcohólicas por continente')
plt.legend()

plt.show()
