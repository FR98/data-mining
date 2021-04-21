"""
---------------------------------------------------------------------------------------------------
	Author
	    Francisco Rosal 18676
---------------------------------------------------------------------------------------------------
Basado en:
    https://medium.com/@alcantararosas/k-means-programando-el-algoritmo-desde-cero-en-python-f633b3a1c125
"""

import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


## Data Analysis
data = pandas.read_csv('advertising.csv')
data.head()
data.info()

## Exploratory Data Analysis
print("Exploratory Data Analysis")
# Histograma Edad.
plt.hist(data.Age)
plt.show()

# Crear un jointplot que muestre la relación de la edad vs los ingresos del área.
seaborn.jointplot(data = data, x = 'Age', y = 'Area Income')
plt.show()

# Crear un jointplot que muestre la distribución k de la relación de la edad vs el tiempo que diario empleado.
seaborn.jointplot(data = data, x = 'Age', y = 'Daily Time Spent on Site', color = 'red', kind = 'kde')
plt.show()

# Crear un jointplot del Daily Time Spent on Site vs. Daily Internet Usage.
seaborn.jointplot(data = data, x = 'Daily Time Spent on Site', y = 'Daily Internet Usage')
plt.show()

# Finalmente, crear un pairplot que separe las poblaciones de 'Clicked on Ad'.
seaborn.pairplot(data = data, hue = 'Clicked on Ad', palette = 'bwr')
plt.show()

## Logistic Regression
print("Logistic Regression")
X = data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = data['Clicked on Ad']

# Usar train_test_split para generar X_train, X_test, y_train y y_test con un 33% en el split para split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

# Entrene y ajuste un modelo de regresión logística en el conjunto de entrenamiento.
clf = LogisticRegression(random_state = 0).fit(X_train, y_train)

## Predictions and Evaluations
print("Predictions and Evaluations")
clf.predict(X_test)

# Crear un reporte de classificación (classification_report) para el modelo.
print(classification_report(y_test, clf.predict(X_test)))

# Imprima el confussion matrix de la clasificación.
confusion_matrix(y_test, clf.predict(X_test))

# Aplique la clasificación utilizando Random Forest y compare los resultados.
rfc = RandomForestClassifier(max_depth = 2, random_state = 0).fit(X_train, y_train)
rfc.predict(X_test)
print(classification_report(y_test, rfc.predict(X_test)))
confusion_matrix(y_test, rfc.predict(X_test))
