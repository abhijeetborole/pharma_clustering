import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import csv
import math
import pandas as pd
x = 0
y = 0
flag = 0
#A = [[0.0 for x in range(20706)] for y in range(23)]
A = np.array([])
#B = np.array([])
B = np.zeros(shape=(20707, 23))

with open('AdSp.txt', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for line in csv_reader:
        if(flag == 0):
            flag = 1
            continue
        for i in [50, 57, 65, 68, 78, 79, 80, 99, 101, 102, 103, 156, 157, 158, 159, 160, 161, 168, 169, 170, 171, 172, 173]:
            try:
                #                A[x][y] = float(line[i].replace(',', ''))
                A = np.append(A, float(line[i].replace(',', '')))
                x += 1
            except ValueError:
                #A[x][y] = 0.0
                A = np.append(A, 0.0)
                x += 1
        B[y] = A
        y += 1
        x = 0
#        B = np.append(B, A, axis=0)
        A = []
print(*B, sep='\n')
print(type(B))
clusterer = hdbscan.HDBSCAN()
clusterer.fit(B)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.savefig('foo.png')
# print(clusterer.labels_)
