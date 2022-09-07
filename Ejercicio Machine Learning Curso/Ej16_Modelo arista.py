#-------------------------------------------------------------------------------
# Name:        Ejercicio 16
# Purpose:      Encontrar dos variables que expliquen la variables stock_volume
# Realizar y componer un modelo Arista, entrenar los datos y obtener R2
# Author:      Miguel Ángel
# Created:     02/04/2022
#-------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pingouin as pg
from statsmodels.graphics.factorplots import interaction_plot
from scipy import stats
from scipy.stats import pearsonr

#Cargamos la base de datos
data = pd.read_csv(r'Nasdaq.csv')
# head: exchange,stock_symbol,date,stock_price_close,stock_gain_loss_dollar,stock_gain_loss_percent,dollar_volume,stock_volume

def describir_base ():
    print("exchange")                              # Exhange es una variable con únicamente el valor NASDAQ, no sirve para análisis
    print(data["exchange"].describe())
    print(pd.unique(data["exchange"]))
    print(" ")
    print("stock_symbol")
    print(data["stock_symbol"].describe())      # stock_symbol: Categórica, 88 valores distintos.
    print(pd.unique(data["stock_symbol"]))
    print(" ")
    print("date")
    print(data["date"].describe())             # date: tipo objetos, 252 unique
    print(" ")
    print("stock_price_close")               # stock_price_close: numérica, tipo float, sin valores de 0 absoluto
    print(data["stock_price_close"].describe())
    print(" ")
    print("stock_gain_loss_dollar")                      # stock_gain_loss_dollar: numérica, tipo float
    print(data["stock_gain_loss_dollar"].describe())
    print(" ")
    print("stock_gain_loss_percent")                     # stock_gain_loss_percent:numérica, tipo float
    print(data["stock_gain_loss_percent"].describe())
    print(" ")
    print("dollar_volume")
    print(data["dollar_volume"].describe())             # dollar_volume:numérica, tipo float
    print(" ")
    print("stock_volume")
    print(data["stock_volume"].describe())           # stock_volume:numérica, tipo float
    print(" ")

# describir_base()

# Esta sería una forma muy rapida de calcularlo, pero el interprete de PyScripter no enseña todos los resultados
def corrpearsonrapido():
    corr_pearson = data.corr(method="pearson")
    print(corr_pearson)
    plt.matshow(data.corr())
    plt.show()

resultado1 = pearson_coef, p_value=stats.pearsonr(data.stock_price_close, data.stock_volume)
resultado2 = pearson_coef, p_value=stats.pearsonr(data.stock_gain_loss_dollar, data.stock_volume)
resultado3 = pearson_coef, p_value=stats.pearsonr(data.stock_gain_loss_percent, data.stock_volume)
resultado4 = pearson_coef, p_value=stats.pearsonr(data.dollar_volume, data.stock_volume)

print(resultado1)
print(resultado2)
print(resultado3)
print(resultado4)


def graficar_numericas():

        fig, axs =plt.subplots(nrows=2, ncols=2)

        axs[0,0].scatter(data.stock_volume, data.stock_price_close)
        axs[0,0].set_title('Stock price close')

        axs[0,1].scatter(data.stock_volume, data.stock_gain_loss_dollar)
        axs[0, 1].set_title('Stock gain loss dollar')

        axs[1,0].scatter(data.stock_volume, data.stock_gain_loss_percent)
        axs[1, 0].set_title('Stock gain loss percent')

        axs[1,1].scatter(data.stock_volume, data.dollar_volume)
        axs[1, 1].set_title('Dollar Volume')
        plt.show()


graficar_numericas()

print ("Con los valores obtenidos por la correlacion de Pearson y la graficacion de los datos, empleamos stock_gain_loss dollar y dollar volume")


# Intentar Refinar con el Modelo Arista.

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


y_data=data['stock_volume']
x_data=data[['stock_gain_loss_dollar','dollar_volume']]
pr=PolynomialFeatures(degree=7)
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.3,random_state=0)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
Model_arista=Ridge(alpha=0.55)
Model_arista.fit(x_train_pr, y_train)
yhat = Model_arista.predict(x_train_pr)

print(yhat)
print('Valore predictivos:', yhat[0:4])
print('Test Prueba       :', y_test[0:4].values)
print(Model_arista.score(x_train_pr,y_train))


