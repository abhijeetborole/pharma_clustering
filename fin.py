import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import csv
import math
import pandas as pd
sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}
x = 0
y = 0
dcl = 0
flag = 0
my_dict = {}
B = np.array([])
C = []
with open('AdSp.txt', encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter="\t")
    for line in csv_reader:
        x += 1
        if(x == 1):
            continue
        if(flag == 0):
            for i in range(176):
                try:
                    B = np.append(B, float(line[i].replace(',', '')))
                    C = np.append(C, y)
                    y += 1
                    flag = 1
                except ValueError:
                    y += 1
                    continue
#        print(*C, sep="\n")
        for j in C:
            if(dcl == 0):
                A = np.empty(len(C), float)
                dcl = 1
            try:
                B = np.append(B, float(line[int(j)].replace(',', '')))
            except ValueError:
                B = np.append(B, 0.0)
                continue
        A = np.hstack([A, B])
#        print(len(B))
        B = []
#        print(len(A))
#plt.scatter(A.T[0], A.T[1], color='b', **plot_kwds)

print(len(A))
print(*A, sep="\n")
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
pd.DataFrame(A).head()
clusterer.fit(A)
clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
                                      edge_alpha=0.6,
                                      node_size=80,
                                      edge_linewidth=2)
clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
clusterer.condensed_tree_.plot()
clusterer.condensed_tree_.plot(
    select_clusters=True, selection_palette=sns.color_palette())
palette = sns.color_palette()
cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
plt.savefig('foo.png')
