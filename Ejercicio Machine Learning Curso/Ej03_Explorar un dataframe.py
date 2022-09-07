import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cabeceras
Cabeceras=['Symboling','Normalized-Losses','Make','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
#Cargamos los datos
data = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)


##### 1 #####
print('modelos de autos\n')
marcas = data['Make']
print(marcas.unique())

##### 2 #####
# Consumo de carburante en ciudad -> city-mpg
print('\n consumo de carburante de cada modelo \n ')
conscarb = data.groupby('Make')['city-mpg']
# Convertir city-mpg a float
data["city-mpg"]=data["city-mpg"].astype("float64")
print(conscarb.describe())

##### 3 #####
# Renombramos las variables
data.rename(columns={"city-mpg" : "Consumo-Ciudad"},inplace=True)
# Convertir de galones a litros.
data["Consumo-Ciudad"]=235/data["Consumo-Ciudad"]


##### 4 #####
# Calcular el precio por modelo
print('\nsacar el precio de cada modelo\n')

# Primero Sustiruir ? por '0'
data.loc[data['Price'] == '?', ['Price']] = '0'

# Convertir el precio a Float64
data["Price"]=data["Price"].astype("float64")
data.rename(columns={"Price" : "Precio"},inplace=True)
preciomarca = data.groupby('Make')['Precio']
print(preciomarca.describe())