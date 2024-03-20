# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
data = pd.read_csv('shopping_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Extract relevant columns for clustering
data1 = data.iloc[:, 3:5].values

# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Customer Dendrograms")

# Perform hierarchical clustering
dendrogram = shc.dendrogram(shc.linkage(data1, method='ward'))

# Fit Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=5, linkage='ward')
cluster_labels = cluster.fit_predict(data1)
print(cluster_labels)

# Visualize clustered data
plt.figure(figsize=(10, 7))
plt.scatter(data1[:,0], data1[:,1], c=cluster_labels, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clustered Data')
plt.show()
