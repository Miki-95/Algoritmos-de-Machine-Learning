#-------------------------------------------------------------------------------
# Name:        Ejercicio 15
# Purpose:      Comparar con ANOVA AAME y AACC
# Author:      Miguel Ángel
# Created:     24/03/2022
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pingouin as pg
from statsmodels.graphics.factorplots import interaction_plot

#Cargamos la base de datos
data = pd.read_csv(r'Nasdaq.csv')
# head: exchange,stock_symbol,date,stock_price_close,stock_gain_loss_dollar,stock_gain_loss_percent,dollar_volume,stock_volume

def describir_base ():
    print("exchange")
    print(data["exchange"].describe())
    print(pd.unique(data["exchange"]))
    print(" ")
    print("stock_symbol")
    print(data["stock_symbol"].describe())
    print(pd.unique(data["stock_symbol"]))
    print(" ")
    print("date")
    print(data["date"].describe())
    print(" ")
    print("stock_price_close")
    print(data["stock_price_close"].describe())
    print(" ")
    print("stock_gain_loss_dollar")
    print(data["stock_gain_loss_dollar"].describe())
    print(" ")
    print("stock_gain_loss_percent")
    print(data["stock_gain_loss_percent"].describe())
    print(" ")
    print("dollar_volume")
    print(data["dollar_volume"].describe())
    print(" ")
    print("stock_volume")
    print(data["stock_volume"].describe())
    print(" ")

# describir_base()

# Filtrar la base de datos
data2 = data[["stock_symbol","stock_price_close"]]
data3 = pd.concat([data2[data2['stock_symbol']=='AAME'],data2[data2['stock_symbol']=='AACC']])


#-------------------------------------------------------------------------------
#BOXPLOT

def BOXPLOT ():

    plt.subplot(1, 2, 1)
    ax1 = sns.boxplot(x="stock_symbol", y="stock_price_close", data=data3)
    # Si queremos ver la distribucion de los valores
    plt.title("Sin distribucion")
    plt.ylabel("Valor de cierre")
    plt.xlabel("Empresa")
    plt.show()

    plt.subplot(1, 2, 2)
    ax = sns.boxplot(x="stock_symbol", y="stock_price_close", data=data3)
    # Si queremos ver la distribucion de los valores
    ax2 = sns.swarmplot(x="stock_symbol", y="stock_price_close", data=data3, color=".25")
    plt.title("Con distribucion")
    plt.ylabel("Valor de cierre")
    plt.xlabel("Empresa")

    # Establecer título
    plt.suptitle("Valores de cierre")
    plt.show()

BOXPLOT ()


#-------------------------------------------------------------------------------
#Verificar condiciones para un ANOVA
# Independencia? -> Sí
# Distribución normal de las observaciones


# Gráficos qqplot

def gqqplot():
    fig, axs = plt.subplots(2,2, figsize=(8, 7))
    pg.qqplot(data3.loc[data3.stock_symbol=='AAME', 'stock_price_close'], dist='norm', ax=axs[0,0])
    axs[0,0].set_title('AAME')
    pg.qqplot(data3.loc[data3['stock_symbol']=='AACC', 'stock_price_close'], dist='norm', ax=axs[0,1])
    axs[0,1].set_title('AACC')
    plt.show()
# gqqplot()

# Test de normalidad Shapiro-Wilk
Shapiro = pg.normality(data=data3, dv='stock_price_close', group='stock_symbol')
print("Shapiro")
print(Shapiro)

# Ni el qqplot ni el test de Shapiro apoyan una distribcion normal de los datos. Por lo tanto, no sería lo más adecuado
# hacer un Analisis Anova, de todas formas, seguimos adelante

# Varianza constante entre grupos (homocedasticidad)
Levene = pg.homoscedasticity(data=data3, dv='stock_price_close', group='stock_symbol', method='levene')
print("Levene")
print(Levene)

# Tampoco hay igualdas de varianzas

#-------------------------------------------------------------------------------
# ANOVA
ANOVA= pg.anova(data=data3, dv= 'stock_price_close', between='stock_symbol',detailed=True)
print("ANOVA")
print (ANOVA)

print( "El valor F es muy bajo, por lo que nos daría un resultado estadísitcamente significativo, \n además, el valor de la eta cuadrado (np2) es del 0.87; es decir, la v. independiente explica el 87% de la variabilidad de la v. dependiente")

