#-------------------------------------------------------------------------------
# Name:        Ejercicio Nº 19. Modelos Lineal Múltiple
#              Modelo Lineal Polinómico Múltiple.
#              Conversión Variables categóricas vs Cuantitativas.
#              Evalución y refinamiento modelo.
#              Modelo Arista.
#              Modelo KNN- Vecino Más próximo.
# Author:      Curso Ciencia de Datos
# Created:     10/03/2022
#-------------------------------------------------------------------------------
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics

import pydotplus
from six import StringIO

def main():
  # 2: CARGAR LA BASE DE DATOS
  df_pacientes = pd.read_csv(r'drug200.csv')

  # Las etiquetas en español
  df_pacientes.rename(columns ={"Drug": "Droga"}, inplace = True)
  df_pacientes.rename(columns={"Age" : "Edad"},inplace=True)
  df_pacientes.rename(columns={"Sex" : "Sexo"},inplace=True)
  df_pacientes.rename(columns={"BP" : "Presion-Sanguinea"},inplace=True)
  df_pacientes.rename(columns={"Cholesterol" : "Colesterol"},inplace=True)

  df_pacientes.loc[df_pacientes['Colesterol']=='NORMAL', ['Colesterol']]= 'Normal'
  df_pacientes.loc[df_pacientes['Colesterol']=='HIGH', ['Colesterol']]= 'Alto'

  df_pacientes.loc[df_pacientes['Droga']=='drugY', ['Droga']]= 'DrogaY'
  df_pacientes.loc[df_pacientes['Droga']=='drugC', ['Droga']]= 'DrogaC'
  df_pacientes.loc[df_pacientes['Droga']=='drugA', ['Droga']]= 'DrogaA'
  df_pacientes.loc[df_pacientes['Droga']=='drugX', ['Droga']]= 'DrogaX'

  #3: GENERAMOS LA VARIABLE:
  XXXXX = df_pacientes[['Edad', 'Sexo', 'Presion-Sanguinea', 'Colesterol', 'Na_to_K']].values
  yyyyy = df_pacientes["Droga"]
  
  # El arbol de decisiones no sabe trabajar con valores categoricos
  # REETIQUETAR LOS VALORES CATEGORICOS: colesterol y sexo

  le_sex = preprocessing.LabelEncoder()
  le_sex.fit(['F','M'])
  XXXXX[:,1] = le_sex.transform(XXXXX[:,1])
  le_BP = preprocessing.LabelEncoder()
  le_BP.fit([ 'LOW', 'Normal', 'Alto'])
  XXXXX[:,2] = le_BP.transform(XXXXX[:,2])
  le_Chol = preprocessing.LabelEncoder()
  le_Chol.fit([ 'Normal', 'Alto'])
  XXXXX[:,3] = le_Chol.transform(XXXXX[:,3])


  # 4: CREAMOS EL ARBOL
  X_trainset, X_testset, y_trainset, y_testset = train_test_split(XXXXX, yyyyy, test_size=0.3, random_state=0.3)
  
  # Primero crearemos una instancia del DecisionTreeClassifier
  # llamada drugTree. Dentro del clasificador, especificaremos
  # criterion="entropy" para que podamos ver la nueva información de cada nodo.
  drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
  
  # Luego, adaptaremos los datos con la matriz de entrenamiento X_trainset
  # y el vector de respuesta y_trainset.
  drugTree.fit(X_trainset,y_trainset)
  predTree = drugTree.predict(X_testset)
  
  # revisemos la precisión de nuestro modelo.
  PrecisionModelo=metrics.accuracy_score(y_testset, predTree)
  print("Precisión de mi Modelo Clasificación Arbol de Decisión: ",PrecisionModelo )
  if (PrecisionModelo>0.55):
    print("La precisión del Modelo es adecuada.")
  else:
    print("La precisión del modelo es mala o el Modelo es erróneo.Verifique el Algoritmo")

  #Diseñar Gráfico de Esquema.
  #Observemos el árbol
  dot_data = StringIO()
  filename = "esquema.png"
  featureNames = df_pacientes.columns[0:5]
  targetNames = df_pacientes["Droga"].unique().tolist()
  out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
  graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
  graph.write_png(filename)
  img = mpimg.imread(filename)
  plt.figure(figsize=(100, 200))
  plt.imshow(img,interpolation='nearest')
  plt.show()
main()

