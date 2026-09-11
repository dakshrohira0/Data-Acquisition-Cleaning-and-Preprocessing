import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set plot style
sns.set_theme(style='whitegrid')

# 1. Load the cleaned data
df = pd.read_csv('cleaned_housing_dataset.csv')
print('Data loaded successfully! Shape:', df.shape)

# 2. Select features for clustering
features = ['Square_Footage', 'Price', 'Price_Per_Sqft', 'Property_Age']
X = df[features]

# Scale features using standard scaling (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find optimal K using Elbow Method & Silhouette Score
wcss = []
silhouette_list = []
k_values = range(2, 9)

for k in k_values:
  km = KMeans(n_clusters=k, random_state=42, n_init=10)
  km.fit(X_scaled)
  wcss.append(km.inertia_)
  silhouette_list.append(silhouette_score(X_scaled, km.labels_))

# Plot Elbow and Silhouette curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

axes[0].plot(k_values, wcss, marker='o', color='blue')
axes[0].set_title('Elbow Method (WCSS)')
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia / WCSS')
axes[0].axvline(x=3, color='red', linestyle='--', label='Elbow Point (k=3)')
axes[0].legend()

axes[1].plot(k_values, silhouette_list, marker='s', color='green')
axes[1].set_title('Silhouette Score across K')
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].axvline(x=3, color='red', linestyle='--', label='Selected k=3')
axes[1].legend()

plt.tight_layout()
plt.show()

# 4. Fit final K-Means model with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Assign readable cluster names based on means
cluster_names = {
    0: 'Modern Starter Homes',
    1: 'Premium Estates',
    2: 'Mature Historic Homes',
}
df['Cluster_Name'] = df['Cluster'].map(cluster_names)

# Print average stats for each cluster
print('\nCluster Summary (Average Values):')
print(
    df.groupby('Cluster_Name')[
        ['Square_Footage', 'Price', 'Price_Per_Sqft', 'Property_Age']
    ].mean()
)

# 5. Scatter plot of Clusters (Square Footage vs Listing Price)
plt.figure(figsize=(8.5, 5))
sns.scatterplot(
    x='Square_Footage',
    y='Price',
    hue='Cluster_Name',
    data=df,
    palette='Set1',
    alpha=0.8,
)

# Mark centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c='black',
    s=150,
    marker='X',
    label='Centroids',
)

plt.title('K-Means Property Clusters (k=3)')
plt.xlabel('Square Footage (sq ft)')
plt.ylabel('Listing Price ($)')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Hierarchical Clustering Dendrogram (Validation)
# Sample 50 rows for clean dendrogram visualization
sample_data = X_scaled[np.random.choice(len(X_scaled), 50, replace=False)]
link = linkage(sample_data, method='ward')

plt.figure(figsize=(9, 4.5))
dendrogram(link, truncate_mode='lastp', p=20)
plt.axhline(
    y=7.0, color='red', linestyle='--', label='Cut Threshold (3 Clusters)'
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Sample Groups')
plt.ylabel('Distance')
plt.legend()
plt.tight_layout()
plt.show()

# 7. Save clustered data to CSV
df.to_csv('clustered_housing_dataset.csv', index=False)
print('Done! Clustered dataset saved to clustered_housing_dataset.csv')
