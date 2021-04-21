"""
---------------------------------------------------------------------------------------------------
	Author
	    Francisco Rosal 18676
---------------------------------------------------------------------------------------------------
"""

import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


## Exploración de datos
data = pandas.read_csv('insurance.csv')
# age,sex,bmi,children,smoker,region,charges
# data.head()
# data.info()


# Utilizando las diferentes librerías que ya conoce,
# realice una exploración de los datos guiándose(más no limitándose)
# por las siguientes tareas:
# -Explore la información del dataset: estructura e información de los campos que contiene.
# -Visualice los campos numéricos en una gráfica de pairwise donde se pueda apreciar la distribución de los mismos.
# -Para los campos no numéricos puede graficar histogramas para conocer su distribución, o incluirlos en los pairwise con claves de color.

# seaborn.pairplot(data = data[['bmi', 'children', 'smoker', 'charges']], palette = 'bwr')
# plt.show()

# plt.hist(data.age)
# plt.show()

# plt.hist(data.region)
# plt.show()

# plt.hist(data.sex)
# plt.show()

# No hay datos nulos en el dataset

## Datos categoricos

# Dado que estaremos aplicando una regresión lineal, es necesario que codifiquemos nuestros datos categóricos.
# Existen dos tipos de codificaciónque seguramente se incluyen en la librería que esté utilizando según
# el lenguaje y herramienta con el que esté trabajando. Estos son: Label Encoding y On Hot Encoding;
# el primero para codificaciones binarias y el segundo para cuando la variable puede tener tres o más valores.
# Utilice un Label Encoder para codificar los campos de sexy smoker, y el On Hot Encoder para el delregion.

label_encoding = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder()

label_encoding.fit(data['sex'])
label_encoding.fit(data['smoker'])

one_hot_encoder.fit(data[['region']])


## Dividir en training y test

# Vamos a dividir nuestro dataset en training y test. La variabledependiente que vamos a querer predecir es el campo charges.
# data_X = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
data_X = data.drop(columns = ['charges'], axis=1)
data_Y = data.charges

x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.33)


## Preparacion de los datos - escala
# scaler = preprocessing.StandardScaler()
# scaler.fit(data)
# x_train, x_test = scaler.fit_transform(x_train), scaler.fit_transform(x_test)

regression = LinearRegression().fit(X = x_train, y = y_train)
y_pred = regression.predict(x_test)

# plt.scatter(x_test, y_test)
# plt.plot(x_test, y_pred, color = 'red')

# plt.xticks(())
# plt.yticks(())
# plt.show()



# print(classification_report(y_test, regression.predict(x_test)))
# confusion_matrix(y_test, regression.predict(x_test))

# # Aplique la clasificación utilizando Random Forest y compare los resultados.
# rfc = RandomForestClassifier(max_depth = 2, random_state = 0).fit(X_train, y_train)
# rfc.predict(X_test)
# print(classification_report(y_test, rfc.predict(X_test)))
# confusion_matrix(y_test, rfc.predict(X_test))
