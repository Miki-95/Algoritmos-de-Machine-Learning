#-------------------------------------------------------------------------------
# Name:         Ejercicio 17
# Purpose:      Buscar un valor de K
# Author:      Miguel Ángel
# Created:     02/04/2022
#-------------------------------------------------------------------------------

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv(r'telecust1000t.csv')

data.describe()

# Generamos las variables
X = data[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
# Normalizar Datos. Todos los datos de un mismo Tipo.
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
y = data['custcat'].values

# Seleccionamos un subconjunto de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

# Calcular Certeza y Valor de K.
Ks = 100
mean_acc = np.zeros((Ks-1)) # Crear Lista de Ceros 0.0
std_acc = np.zeros((Ks-1))  # Crear Lista de Ceros 0.0

ConfustionMx = []
for n in range(1,Ks):
    #Entrenar el Modelo y Predecir
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]= metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),mean_acc,'r') # Trazado Lineal de 1 a 10 Línea Roja.
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10) # Rellena Área entre 2 líneas Horizontales.alpha tipo Matriz escalar o ninguno
plt.legend(('Certeza ', '+/- 3xstd'))
plt.ylabel('Certeza ')
plt.xlabel('Número de Vecinos (K)')
# Ajuste el relleno entre y alrededor de las subparcelas.
plt.tight_layout()
plt.show()

# argmax() .Devuelve los índices de los valores máximos a lo largo de un eje
print( "La mejor aproximación de certeza fue con ", mean_acc.max(), "con k=", mean_acc.argmax()+1)

# Entrenar el Modelo y Predecir
k = mean_acc.argmax()+1
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print(neigh)
yhat = neigh.predict(X_test)
print(yhat[0:5])

# Gráfico de Distribución que nos visualize la diferencia entre Valor Real y Predictivo.
axl=sns.distplot(data['custcat'],hist=False,color='r', label='Valor Actual')
sns.distplot(yhat,hist=False,color='b', label='Valores Ajustados',ax=axl)
plt.ylim(0,) #Obtener o establecer los límites y de los ejes actuales.
plt.show()
plt.close()