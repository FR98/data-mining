"""
---------------------------------------------------------------------------------------------------
	Author
	    Francisco Rosal 18676
---------------------------------------------------------------------------------------------------
Basado en:
    https://medium.com/@alcantararosas/k-means-programando-el-algoritmo-desde-cero-en-python-f633b3a1c125
"""

import pandas
import numpy
import matplotlib.pyplot as plt

data = pandas.read_csv('iris.csv')
data_size = len(data)
print(data.info())
K = 3

sepal_length_min, sepal_length_max = numpy.min(data.sepal_length) * 10, numpy.max(data.sepal_length) * 10
sepal_width_min, sepal_width_max = numpy.min(data.sepal_width) * 10, numpy.max(data.sepal_width) * 10
petal_length_min, petal_length_max = numpy.min(data.petal_length) * 10, numpy.max(data.petal_length) * 10
petal_width_min, petal_width_max = numpy.min(data.petal_width) * 10, numpy.max(data.petal_width) * 10

centroides = []
for i in range(K):
    random_sepal_length = numpy.random.randint(sepal_length_min, sepal_length_max)
    random_sepal_width = numpy.random.randint(sepal_width_min, sepal_width_max)
    random_petal_length = numpy.random.randint(petal_length_min, petal_length_max)
    random_petal_width = numpy.random.randint(petal_width_min, petal_width_max)
    centroides.append([random_sepal_length/10, random_sepal_width/10, random_petal_length/10, random_petal_width/10])

r = numpy.zeros((data_size, K))

for i in range(10):
    d_euclidiana = 0

    # Obtenemos las distancias de todos los puntos a cada centroide
    for n in range(data_size):
        point = numpy.array(data.iloc[n, 0:4])
        mins = [pow(numpy.linalg.norm(point - centroides[k]), 2) for k in range(K)]
        # Se decide cual es la distancia min
        r[n, numpy.argmin(numpy.array(mins))] = 1

    # Obtenemos distancia euclidiana
    for n in range(data_size):
        for k in range(K):
            d_euclidiana = r[n, k] * pow(numpy.linalg.norm(point - centroides[k]), 2) + d_euclidiana

    # Movemos los centroides
    for k in range(K):
        count_1, count_2 = 0, 0
        for n in range(data_size):
            point = numpy.array(data.iloc[n, 0:4])
            count_1 = r[n, k] * point + count_1
            count_2 = r[n, k] + count_2
        centroides[k] = count_1 / count_2

    if i > 1 and abs(d_euclidiana - d_euclidiana_prev) < 0.5: break
    d_euclidiana_prev = d_euclidiana
    r = numpy.zeros((data_size, K))

group = [numpy.argmax(row) + 1 for row in r]

centroides = pandas.DataFrame(centroides)
plt.scatter(data.sepal_length, data.sepal_width, c=group, marker=".")
plt.scatter(centroides[0], centroides[1], marker="+", color="red", s=100)
plt.show()

centroides = pandas.DataFrame(centroides)
plt.scatter(data.petal_length, data.petal_width, c=group, marker=".")
plt.scatter(centroides[2], centroides[3], marker="+", color="red", s=100)
plt.show()
