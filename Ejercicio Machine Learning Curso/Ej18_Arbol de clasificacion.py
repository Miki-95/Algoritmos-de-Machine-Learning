#-------------------------------------------------------------------------------
# Name:        Ejercicio 18
# Purpose:     Realizar un arbol de decisiones con
#               V. independientes: Edad, colesterol y sexo
#               V. dependiente: Medicamento
# Author:       Miguel Ángel
# Created:     10/03/2022
#-------------------------------------------------------------------------------
# 1: CARGAR LAS LIBRERIAS
import pandas as pd  # Manipulacion de datos
import numpy as np

import matplotlib.pyplot as plt # Visualizacion de modelos
import matplotlib.image as mpimg

from sklearn import metrics
from sklearn import preprocessing # Libreria de machine learning
from sklearn.model_selection import train_test_split # Libreria para educar a modelos
from sklearn.tree import DecisionTreeClassifier # Arbol de decisiones
from sklearn import tree

import pydotplus

from six import StringIO

def main():
    #-------------------------------------------------------------------------------
    # 2: CARGAR LA BASE DE DATOS

    df_pacientes = pd.read_csv(r"drug200.csv")
    df_pacientes[0:5]
    df_pacientes.size

    # Las etiquetas en español
    df_pacientes.rename(columns ={"Drug": "Droga"}, inplace = True)
    df_pacientes.rename(columns ={"Age": "Edad"}, inplace = True)
    df_pacientes.rename(columns ={"Sex": "Sexo"}, inplace = True)
    df_pacientes.rename(columns ={"Cholesterol": "Colesterol"}, inplace = True)

    df_pacientes.loc[df_pacientes['Colesterol']=='NORMAL', ['Colesterol']]= 'Normal'
    df_pacientes.loc[df_pacientes['Colesterol']=='HIGH', ['Colesterol']]= 'Alto'

    df_pacientes.loc[df_pacientes['Droga']=='drugY', ['Droga']]= 'DrogaY'
    df_pacientes.loc[df_pacientes['Droga']=='drugC', ['Droga']]= 'DrogaC'
    df_pacientes.loc[df_pacientes['Droga']=='drugA', ['Droga']]= 'DrogaA'
    df_pacientes.loc[df_pacientes['Droga']=='drugX', ['Droga']]= 'DrogaX'
    #-------------------------------------------------------------------------------

    # 3: GENERAMOS LA VARIABLE INDEPENDIENTE: Edad, colesterol y sexo
    X = df_pacientes[['Edad', 'Sexo',  'Colesterol']].values

    # El arbol de decisiones no sabe trabajar con valores categoricos
    # REETIQUETAR LOS VALORES CATEGORICOS: colesterol y sexo

    # Etiquetar sexo
    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F','M'])
    X[:,1] = le_sex.transform(X[:,1])

    # Etiquetar colesterol
    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit([ 'Normal', 'Alto'])
    X[:,2] = le_Chol.transform(X[:,2])
    X[0:5]
    #-------------------------------------------------------------------------------
    # 4: GENERAMOS LA VARIABLE DEPENDIENTE

    y = df_pacientes["Droga"]
    #-------------------------------------------------------------------------------

    # 5: ENTRENAMOS EL MODELO
    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
    #-------------------------------------------------------------------------------
    # 6: CREAMOS EL ARBOL

    # Primero crearemos una instancia del DecisionTreeClassifier
    # llamada drugTree. Dentro del clasificador, especificaremos
    # criterion="entropy" para que podamos ver la nueva información de cada nodo.
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

    # Luego, adaptaremos los datos con la matriz de entrenamiento X_trainset
    # y el vector de respuesta y_trainset.
    drugTree.fit(X_trainset,y_trainset)
    predTree = drugTree.predict(X_testset)
    print (predTree [0:5])
    print (y_testset [0:5])

    # revisemos la precisión de nuestro modelo.
    print("Precisión de los Arboles de Decisión: ", metrics.accuracy_score(y_testset, predTree))
    #-------------------------------------------------------------------------------
    # 7: VISUALIZACION DEL ARBOL

    dot_data = StringIO()
    filename = "drugtree.png"
    featureNames = df_pacientes.columns[0:3]
    targetNames = df_pacientes["Droga"].unique().tolist()
    out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(filename)
    img = mpimg.imread(filename)
    plt.figure(figsize=(100, 200))
    plt.imshow(img,interpolation='nearest')
    plt.show()

main()