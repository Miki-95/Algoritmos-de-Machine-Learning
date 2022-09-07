#-------------------------------------------------------------------------------
# Purpose:
# Author:      Miguel Ángel
# Created:     27/01/2022
# Copyright:   (c) Miguel Ángel 2022
#-------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Leemos la base de datos drinks
data = pd.read_csv(r"drinks.csv", header=0)

#print(data.shape)
#print (data.head(10))

# BOXPLOT
# Visualozamos los datos agrupados por continente
print(data.groupby(['continent']).sum())

# Dibujamos un boxplot con el consumo de porciones de vino por continente
boxplot = data.boxplot(column=['wine_servings'],by="continent")
boxplot.plot()
plt.show()

# REGRESION
sns.regplot(x = "wine_servings",y = "beer_servings",data = data,
 dropna = True,scatter_kws={"color": "black"}, line_kws={"color":
"red"})
plt.show()