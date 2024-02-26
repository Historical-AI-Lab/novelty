
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from k_means_constrained import KMeansConstrained
import math, sys

# Get the input filename from sys.argv
input_filename = sys.argv[1]

# Read a single line of the file to infer the number of columns
with open(input_filename, 'r') as file:
    line = file.readline()
    num_columns = len(line.split('\t'))

# Define the column names
column_names = ['chunkid'] + [f'V{i}' for i in range(1, num_columns)]

# Read the first 2000 lines of the file into a dataframe
df = pd.read_csv(input_filename, sep='\t', header=None, names=column_names)


df[['articleid', 'chunkseq']] = df['chunkid'].str.split('-', expand=True)

def write_centroids(articleid, centroids, filename):
    with open(filename, 'a') as file:
        for i, centroid in enumerate(centroids):
            chunkid = f'{articleid}-{i}'
            file.write(chunkid + '\t' + '\t'.join([str(x) for x in centroid]) + '\n')

# Group the dataframe by articleid
groups = df.groupby('articleid')

# Initialize a counter variable
counter = 0

# Iterate through the groups
for articleid, group in groups:
    # Get the numeric matrix
    matrix = group.iloc[:, 1:769].values
    
    # Perform PCA on the matrix
    pca = PCA(n_components=0.7)
    transformed_matrix = pca.fit_transform(matrix)
    
    # Print the number of dimensions retained
    # print(f"Article ID: {articleid}")
    # print(f"Number of dimensions retained: {pca.n_components_}")
    
    # Print the number of rows in the matrix
    nrows = matrix.shape[0]
    if nrows < 3:
        continue    # skip articles with less than 3 sentences
    
    # Increment the counter
    counter += 1
    if counter % 200 == 0:
        print(f"Processed {counter} articles")

    num_clusters = max(math.ceil(nrows / 15), 1)

    # Calculate the average number of rows per cluster
    avg_rows_per_cluster = round(nrows / num_clusters)

    # Set the minimum and maximum number of rows per cluster
    min_rows_per_cluster = avg_rows_per_cluster - 2
    max_rows_per_cluster = avg_rows_per_cluster + 2

    centroids = []
    if num_clusters == 1:
        # only one cluster, no need to run kmeans
        # calculate the centroid of the matrix
        centroids.append(np.mean(matrix, axis=0))
        write_centroids(articleid, centroids, 'cluster_centroids.tsv')
        continue

    # Initialize the KMeansConstrained model
    kmeans = KMeansConstrained(
        n_clusters=num_clusters,
        size_min=min_rows_per_cluster,
        size_max=max_rows_per_cluster,
        random_state=0
    )
    # Fit the model
    kmeans.fit(transformed_matrix)
    # print(kmeans.labels_)

    for cluster_id in range(num_clusters):
        # Get the indexes of rows assigned to the current cluster_id
        cluster_indexes = np.where(kmeans.labels_ == cluster_id)[0]
        
        # Get the vectors from the original matrix
        cluster_vectors = matrix[cluster_indexes]
        
        # Calculate the centroid for the cluster
        centroid = np.mean(cluster_vectors, axis=0)
        
        # Append the centroid to the centroids list
        centroids.append(centroid)

    # Write the centroids to a file
    write_centroids(articleid, centroids, 'cluster_centroids.tsv')

print('Done.')