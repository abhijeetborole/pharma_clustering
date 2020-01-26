import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import csv
import pandas as pd
np.set_printoptions(threshold=np.nan)
if __name__ == "__main__":
    sns.set_context('poster')
    sns.set_style('white')
    sns.set_color_codes()
    palette = sns.color_palette()
    plot_kwds = {'alpha': 0.5, 's': 80, 'linewidths': 0}
    x = 0
    y = 0
    flag = 0
    A = np.array([])
    B = np.zeros(shape=(20707, 23))
    with open('AdSp.txt', encoding='utf8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        for line in csv_reader:
            if(flag == 0):
                flag = 1
                continue
            for i in [50, 57, 65, 68, 78, 79, 80, 99, 101, 102, 103, 156, 157, 158, 159, 160, 161, 168, 169, 170, 171, 172, 173]:
                try:
                    A = np.append(A, float(line[i].replace(',', '')))
                    x += 1
                except ValueError:
                    A = np.append(A, 0.0)
                    x += 1
            B[y] = A
            y += 1
            x = 0
            A = []
    #print(*B, sep='\n')
    # print(type(B))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=75)
    clusterer.fit(B)
    # print(clusterer.outlier_scores_)
    print(max(clusterer.labels_))
    # print(clusterer.labels_)
    p = TSNE(n_components=2).fit_transform(B)
    plt.scatter(*p.T, s=50, linewidth=0, c='b', alpha=0.25)
    plt.savefig('plot.png')
    # clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #                                      edge_alpha=0.6,
    #                                      node_size=80,
    #                                      edge_linewidth=2)
    # plt.savefig('minimum_spanning_tree.png') //No Minimum Spanning Tree generated as algorithm skips this for better optimization in some variations
    #clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    # plt.savefig('single_linkage_tree.png')
    clusterer.condensed_tree_.plot(
        select_clusters=True, selection_palette=sns.color_palette())
    plt.savefig('condensed_tree_75.png')
    sns.distplot(clusterer.outlier_scores_[
        np.isfinite(clusterer.outlier_scores_)], rug=True)
    threshold = pd.Series(clusterer.outlier_scores_).quantile(0.95)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    #color_palette = sns.color_palette('Paired', 13)
    # cluster_colors = [color_palette[x] if x >= 0
    #                  else (0.5, 0.5, 0.5)
    #                  for x in clusterer.labels_]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                         zip(cluster_colors, clusterer.probabilities_)]
    # plt.scatter(*p.T, s=50, linewidth=0,
    #            c=cluster_member_colors, alpha=0.25)
    plt.scatter(*p.T, s=50, linewidth=0, c='gray', alpha=0.25)
    plt.scatter(*p[outliers].T, s=50, linewidth=0, c='red', alpha=0.5)
    plt.savefig('outliers_75.png')
