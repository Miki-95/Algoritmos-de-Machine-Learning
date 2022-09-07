#-------------------------------------------------------------------------------
# Python para Ciencia de Datos

 # 1.Crea un diagrama de dispersión para ver si existe correlación entre el
 #   Precio y el Tamaño del Motor.

 # 2. Realizar con el método de Correlación Pearson si existe correlación o
 #     fortaleza entre RPM y Precio.

 # 3. Realizar un Test para establecer si existe diferencia sustancial entre el
 #     precio Medio de las Marcas de Autos Audi y alfa-romero.

#-------------------------------------------------------------------------------


# CARGAMOS LAS LIBRERIAS QUE NOS HACEN FALTA

# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y análisis
# ==============================================================================
import statsmodels.api as sm
from scipy import stats



### 0 ####

#### Abrir base de datos
# Abrimos el archivo CSV con todos los datos que  necesitamos

# Cabeceras
Cabeceras=['Symboling','Normalized-Losses','Make','Fuel-Type','Aspiration','num-of-doors','Body-Style','Drive-wels','engine-location','whel-base','length','width','height','curb-weight','engine-tpe','num-of-cylinders','engine-zize','fuel-system','bora','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','Price']
#Cargamos los datos
df_autos = pd.read_csv(r"autos.csv", header=None, names = Cabeceras)


#### Depurar datos
# Filtro estructura de valores y dejo aquellos no nulos.
df_autos=df_autos.dropna(subset=['Make','city-mpg','Price', 'engine-zize'],axis=0)
# Sustiruir ? por '0'
df_autos.loc[df_autos['Price'] == '?', ['Price']] = '0'
# Convertir el precio a Float64
df_autos["Price"]=df_autos["Price"].astype("float64")
# Convertir city-mpg a float
df_autos["city-mpg"]=df_autos["city-mpg"].astype("float64")
# Convertir engine-zize a float
df_autos["engine-zize"]=df_autos["engine-zize"].astype("float64")


# renombrar Variables.
df_autos.rename(columns={"city-mpg" : "Consumo-Kilometros"},inplace=True)
# Convertir Galones a lItros 
df_autos["Consumo-Kilometros"]=235/df_autos["Consumo-Kilometros"]

# renombrar Variable Price.
df_autos.rename(columns={"Price" : "Precio"},inplace=True)
# Convertir  precios a Euros (aprox)
df_autos["Precio-Vehiculo"]=df_autos["Precio"] * 0.90



#############################################################################################################################################################################
# OBJETIVO 1: Grafico de dispersion tamaño del motor - precio
#############################################################################################################################################################################


    # Tamaño del motor -> Variable "engine-zize"
    # Precio -> Variable "Precio-Vehiculo"

    # Al intertar quitarnos lo valores de Precio desconocido hemos cambiado su valor a 0
        # Sin embargo, esto añade ruido al estadístico y la fiabilidad de sus resultados.
        # Intentamos crear una variable nueva excluyendo ese 0


# Creamos un nuevo data frame para el ejercicio 01 con las variables que deseamos estudiar
df_Ej01=df_autos[['engine-zize','Precio-Vehiculo']]

# Podemos castellanizar la variable "Tamaño del motor"
df_Ej01.rename(columns={"engine-zize" : "Tamano-Motor"},inplace=True)

# Eliminamos todas las filas que tienen un valor de 0 para el precio
df_Ej01 = df_Ej01.drop(df_Ej01[df_Ej01['Precio-Vehiculo']==0].index)

# df_Ej01.describe() # Por si nos apetece cotillear la E. descriptiva de nuestro dataframe para el Ej. 01



# Definimos los ejes para nuestro gráfico de dispersión. Variable dependiente a la y e independiente a la x
y=df_Ej01['Precio-Vehiculo']
x=df_Ej01['Tamano-Motor']

# Ordenamos la graficacion de nuestros datos
plt.scatter(x,y)
plt.title('Relación Tamaño del Motor con el Precio Vehículo')
plt.xlabel('Tamaño Vehiculo')
plt.ylabel('Precio Vehículo')
sns.regplot(x,y,data=df_Ej01,dropna=True,scatter_kws={"color" : "black"},line_kws={"color" : "blue"})

#Pedimos la graficacion de nuestros datos
plt.show()




#############################################################################################################################################################################

#############################################################################################################################################################################
# OBJETIVO 2: Test de correlacion de Pearson entre el precio y las revoluciones por minuto
#############################################################################################################################################################################


    # Revoluciones por minuto -> Variable "peak-rpm"
    # Precio -> Variable "Precio-Vehiculo"

    # Al intertar quitarnos lo valores de Precio desconocido hemos cambiado su valor a 0
        # Sin embargo, esto añade ruido al estadístico y la fiabilidad de sus resultados.
        # Intentamos crear una variable nueva excluyendo ese 0



# Parece que el rpm está también guardado como un objeto, vamos a convertir los ? en 0

# Filtro estructura de valores y dejo aquellos no nulos.
df_autos=df_autos.dropna(subset=['peak-rpm'],axis=0)
# Sustiruir ? por '0'
df_autos.loc[df_autos['peak-rpm'] == '?', ['peak-rpm']] = '0'
# Convertir el peak-rpm a Float64
df_autos["peak-rpm"]=df_autos["peak-rpm"].astype("float64")

# Creamos un nuevo data frame para el ejercicio 01 con las variables que deseamos estudiar

df_Ej02=df_autos[['peak-rpm','Precio-Vehiculo']]

# Eliminamos todas las filas que tienen un valor de 0 para el precio y el rmp
df_Ej02 = df_Ej02.drop(df_Ej02[df_Ej02['Precio-Vehiculo']==0].index)
df_Ej02 = df_Ej02.drop(df_Ej02[df_Ej02['peak-rpm']==0].index)

# print(df_Ej02.describe()) # Por si nos apetece cotillear la E. descriptiva de nuestro dataframe para el Ej. 01
# print((df_Ej02['peak-rpm']))


# AUNQUE NO SEA EL OBJETIVO 2, VAMOS A HACER UN GRAFICO DE DISPERSION DE LAS VARIABLES QUE QUEREMOS ENFRENTAR
# ESTO NOS AYUDARÁ A ENTENDER SI ES COHERENTE EL RESULTADO DE NUESTRO TEST DE CORRELACION DE SPEARMAN


# Definimos los ejes para nuestro gráfico de dispersión. Variable dependiente a la y e independiente a la x
y=df_Ej02['Precio-Vehiculo']
x=df_Ej02['peak-rpm']

# Ordenamos la graficacion de nuestros datos
plt.scatter(x,y)
plt.title('Relación de las RPM con el Precio Vehículo')
plt.xlabel('peak-rpm')
plt.ylabel('Precio Vehículo')
sns.regplot(x,y,data=df_Ej02,dropna=True,scatter_kws={"color" : "black"},line_kws={"color" : "blue"})

#Pedimos la graficacion de nuestros datos
plt.show()


# CALCULAMOS EL TEST DE CORRELACION DE SPEARMAN CON LOS DATOS SIN NORMALIZAR

# Prepara para establecer Pearson entre 2 variables.

resulT=pearson_coef,p_value=stats.pearsonr(df_Ej02['peak-rpm'],df_Ej02['Precio-Vehiculo'])
p_valor= p_value=stats.pearsonr(df_Ej02['peak-rpm'],df_Ej02['Precio-Vehiculo'])
print("Resultado Pearson",resulT)

if resulT[1]> 0.05:

    print("El test de Spearman da una relación entre las dos variables del orden de ", resulT[0], "; sin embargo, el resultado no es estadísiticamente significativo y por tanto es de poca fiabilidad")

else:

    print("El test de Spearman da una relación entre las dos variables del orden de ", resulT[0], "; además, el resultado no es estadísiticamente significativo y por tanto es de poca fiabilidad")



#############################################################################################################################################################################

#############################################################################################################################################################################
# OBJETIVO 3: Comparar las medias de precio entre Audi y Alfa-Romeo
#############################################################################################################################################################################

# Para realizar una comparacion de medias necesitamos un ANOVA

    # Marca -> Variable "Make"
    # Precio -> Variable "Precio-Vehiculo"


# Creamos un nuevo data frame para el ejercicio 01 con las variables que deseamos estudiar
df_Ej03=df_autos[['Make','Precio-Vehiculo']]

# Eliminamos todas las filas que tienen un valor de 0 para el precio
df_Ej03 = df_Ej03.drop(df_Ej03[df_Ej03['Precio-Vehiculo']==0].index)

# Anova
Grupo_Anova=df_Ej03.groupby(['Make'])
resultado=stats.f_oneway(Grupo_Anova.get_group('audi')['Precio-Vehiculo'],Grupo_Anova.get_group('alfa-romero')['Precio-Vehiculo'])
print("Resultado Nova",resultado)

# Me gustaría visualizar estos datos en un boxplot
df_Ej03_BoxPlot =df_Ej03.drop(df_Ej03[ [i and j  for i,j in zip(df_Ej03.Make != 'audi', df_Ej03.Make != 'alfa-romero')]].index)


# Estadisticas descriptivas con Grafico de Cajas.
Mr=df_Ej03_BoxPlot["Make"].value_counts()
print(Mr)
sns.boxplot(x="Make",y="Precio-Vehiculo",data=df_Ej03_BoxPlot)
plt.show()
