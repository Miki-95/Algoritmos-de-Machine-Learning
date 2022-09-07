#-------------------------------------------------------------------------------
# Name:        Ejercicio Nº 9. Regresión lineal polinómico Simple con drinks
# Author:      Miguel Ángel Gómez Molinero.
# Created:     22/02/2022
#-------------------------------------------------------------------------------


#CARGAR LOS PAQUETES QUE ME HACEN FALTA

# Para abrir y trabajar bases de datos
import pandas as pd                           # Pandas: abrir y trabajar CSV
import numpy as np
import matplotlib.pyplot as plt

# Abrimos el archivo CSV con todos los datos que  necesitamos
df_drinks= pd.read_csv(r"drinks.csv")

# renombro las columnas a mi gusto
df_drinks.rename(columns={"beer_servings": "cerveza"}, inplace = True)
df_drinks.rename(columns={"spirit_servings": "licor"}, inplace = True)
df_drinks.rename(columns={"wine_servings": "vino"}, inplace= True)
df_drinks.rename(columns={"total_litres_of_pure_alcohol":"puro"}, inplace= True)
df_drinks.rename(columns={"continent":"continente"}, inplace= True)
df_drinks.rename(columns={"country":"pais"}, inplace= True)


# Me apetece primero visualizar algunos datos:
df_drinks.describe()
bycontinent = df_drinks.groupby(['continente'])

print(bycontinent["cerveza", "puro"].describe())

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Relacion polinomica entre la cerveza y el alcohol puro

cerveza = np.array(df_drinks["cerveza"])
puro = np.array(df_drinks["puro"])

plt.scatter(cerveza, puro)
plt.xlabel("Cerveza")
plt.ylabel("Puro")
plt.title("Cerveza - Puro")
plt.show()

cZ = np.polyfit(cerveza, puro, 3)
cP = np.poly1d(cZ)
print ("Coeficientes:",cZ)
print("Modelo  :",cP)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Relacion polinomica entre la vino y el alcohol puro

vino = np.array(df_drinks["vino"])

plt.scatter(vino, puro)
plt.xlabel("Vino")
plt.ylabel("Puro")
plt.title("Vino - Puro")
plt.show()

vZ = np.polyfit(vino, puro, 3)
vP = np.poly1d(vZ)
print ("Coeficientes:",vZ)
print("Modelo  :",vP)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Relacion polinomica entre la licor y el alcohol puro

licor = np.array(df_drinks["licor"])

plt.scatter(licor, puro)
plt.xlabel("Licor")
plt.ylabel("Puro")
plt.title("Licor - Puro")
plt.show()

lZ = np.polyfit(licor, puro, 3)
lP = np.poly1d(cZ)
print ("Coeficientes:",lZ)
print("Modelo  :",lP)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Graficamos el modelo

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    
    minimo = int(round(min(independent_variable)-10))
    maximo = int(round(max(independent_variable)+10))
    longitud = len(independent_variable)

    x_new = np.linspace(minimo, maximo, longitud)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polinomio Segun Modelo')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Puro')
    plt.show()
    plt.close()

PlotPolly(cP, cerveza, puro,'Cerveza')
PlotPolly(vP, vino, puro,'Vino')
PlotPolly(lP, licor, puro,'Licor')
