import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cabeceras
Cabeceras=['Symboling','Normalized-Losses','Make','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
#Cargamos los datos
data = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)


##### 1 #####
# Cuantos y cuales modelos tenemos?
print('modelos de autos\n')
marcas = data['Make']
print(marcas.unique())


##### 2 #####
# Cuál es el consumo en carretera medio por marca
# Primero transformamos los datos a float y de galones a litros
data["city-mpg"]=data["city-mpg"].astype("float64")
data.rename(columns={"city-mpg" : "Consumo-ciudad"},inplace=True)
# Convertir Galones a litros.
data["Consumo-ciudad"]=235/data["Consumo-ciudad"]
# Agrupamos el consumo por marca
consumo_marca = data.groupby('Make')['Consumo-ciudad']
print('\n consumo de carburante por modelo \n ')
print(consumo_marca.describe())


##### 3 #####
# Cual es el precio por marca
# Sustiruir ? por '0'
data.loc[data['Price'] == '?', ['Price']] = '0'
# Convertir el precio a Float64
data["Price"]=data["Price"].astype("float64")
# Hacer una conversión aproxiamada de dolares a euros
data["Price"]=0.9*data["Price"] 
# Renombrar el precio
data.rename(columns={"Price" : "Precio"},inplace=True)
# Para tener una visualizacion general de los datos de Price
# print(data['Precio'].describe())


##### 4 #####
# Agrupamos los precios en valores categóricos
# Creamos las  variables categoricas
Categorias = ["Bajo", "Medio", "Alta"]
# Creamos los puntos de corte
cortes = [1, 5000, 10000, 45000]
# Creamos la nueva columna
data['Precio-Vehiculo2'] = pd.cut(data['Precio'],cortes, labels = Categorias)
# print(data['Precio-Vehiculo2'])


##### 5 #####
## Declaramos valores para el eje x
eje_x = Categorias
## Declaramos valores para el eje y
eje_y = data["Precio-Vehiculo2"].value_counts()

## Creamos un gráfico de barras
plt.bar(eje_x, eje_y)

## Leyendas del gráfico
plt.ylabel('Coches disponibles')
plt.xlabel('Rango de precio')

## Título de Gráfica
plt.title('Cantidad de coches disponibles por rango de precio')

## Mostramos Gráfica
plt.show()


