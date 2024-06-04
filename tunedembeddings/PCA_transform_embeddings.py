import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob

# Read all .tsv files in the "models/" folder
file_paths = glob.glob("embeddings/*.tsv")
dataframes = []

# Iterate over each file and read it as a dataframe
for file_path in file_paths:
    df = pd.read_csv(file_path, delimiter='\t')
    dataframes.append(df)

# Concatenate the dataframes along the vertical axis
data = pd.concat(dataframes, axis=0)
print('Data shape:', data.shape)

metadata = data.iloc[:, :2]
data = data.iloc[:, 2:]

# Apply StandardScaler to standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print('Data scaled shape:', data_scaled.shape)

# Perform PCA
n_components = 350
pca = PCA(n_components=n_components)

# Fit PCA on the scaled data and transform it
transformed_data = pca.fit_transform(data_scaled)

# Print the shape of the transformed data
print("Shape of transformed data:", transformed_data.shape)  # Should be (45000, 300)

# Print the proportion of variance retained by the 300 components
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance by each component:", explained_variance_ratio)
print("Total explained variance (proportion) by 300 components:", np.sum(explained_variance_ratio))

transformed_data = pd.DataFrame(transformed_data)

# Reset indices to ensure unique indices before concatenation
metadata.reset_index(drop=True, inplace=True)
transformed_data.reset_index(drop=True, inplace=True)

# Rejoin metadata with transformed_data
transformed_data = pd.concat([metadata, transformed_data], axis=1)
transformed_data.to_csv("tuned_300_columns.tsv", sep='\t', index=False) 
