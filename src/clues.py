from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

def hierarchical_clustering(X, min_clusters, max_clusters, method='single', metric='euclidean', 
                            compute_MDS_switch=True, 
                            # label_names = short_names, 
                            label_names = 'dummy', 
                            scatter_label=True, 
                            label_rotation = 90, 
                            leaf_font_sizes=10, 
                           truncate_mode='level', 
                           D_leaf_colors=None,
                            color_threshold = 0.13, 
                            colormap_viridis =
                                                {'1':'#fde725',
                                                '2':'#5ec962',
                                                '3':'#21918c',
                                                '4':'#3b528b',
                                                '5':'#440154'
                                                }
                           ):
    """
    Function to perform hierarchical clustering on a n by 3 array with different linkage methods,
    visualize the dendrogram, and determine optimal cluster number using Silhouette score.
    
    Args:
    - X (np.array): Input array of shape (n, 3) containing the data points.
    - min_clusters (int): Minimum number of clusters to consider.
    - max_clusters (int): Maximum number of clusters to consider.
    - method (str): Linkage method for hierarchical clustering. Default is 'single'.
                    Options: 'single', 'complete', 'average', 'ward', 'centroid'.
    """
    
    # Perform hierarchical clustering
    Z = linkage(sp.distance.squareform(X), method=method)
    
    # Initialize arrays to store silhouette scores and optimal cluster numbers
    silhouette_scores = []
    optimal_n_clusters = []
    
    # Iterate over cluster numbers and compute Silhouette score
    for n_clusters in range(min_clusters, max_clusters+1):
        labels = fcluster(Z, n_clusters, criterion='maxclust')
        silhouette_scores.append(silhouette_score(X, labels, metric = metric ))
        optimal_n_clusters.append(n_clusters)
    
    # Determine optimal cluster number with highest Silhouette score
    optimal_n_clusters = np.array(optimal_n_clusters)
    optimal_n_clusters = optimal_n_clusters[silhouette_scores.index(max(silhouette_scores))]
    
    # Perform hierarchical clustering with optimal cluster number
    labels = fcluster(Z, optimal_n_clusters, criterion='maxclust')
    
    
    # Plot the results in 3D coordinates with different colors for each cluster
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(121, projection='3d')
    if compute_MDS_switch:
        X_mds = compute_3DMDS(X, n_components=3)
        ax.scatter(X_mds[:, 0], X_mds[:, 1], X_mds[:, 2], c=labels, cmap='viridis')
        if scatter_label: 
            for i,_ in enumerate(X_mds[:,0]):
    #             print(X_mds[:, 0][i], X_mds[:, 1][i], X_mds[:, 2][i], label_names[i])
                ax.text(X_mds[:, 0][i], X_mds[:, 1][i], X_mds[:, 2][i], 
                        label_names[i], alpha = 0.3) # for printing labels

    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
        if scatter_label: 
            for i,_ in enumerate(X[:, 0]):
                ax.text(X[:, 0][i], X[:, 1][i], X[:, 2][i],label_names[i])

    ax.set_title(f'Hierarchical Clustering ({method} Linkage)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Create an empty list to store silhouette scores
    silhouette_scores = []

    # Perform hierarchical clustering for different number of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        silhouette_scores.append(silhouette_score(X, labels))

    # Find the index of maximum silhouette score
    optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters

    print('Optimal Cluster number is: ', optimal_n_clusters)
    # Plot the number of clusters vs Silhouette score
    ax2 = fig.add_subplot(122)
    ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-')
    ax2.set_title(f'Number of Clusters vs Silhouette Score ({method} Linkage)')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.axvline(optimal_n_clusters, color='r', linestyle='--', label='Optimal Number of Clusters')
    ax2.legend()
    
    # Visualize the dendrogram
    fig = plt.figure(figsize=(12, 7))
#     cpalette = ['#fde725','#5ec962','#21918c','#3b528b','#440154']
    cpalette = [val for key, val in colormap_viridis.items()]
    hierarchy.set_link_color_palette(cpalette[::-1])
    dendrogram(Z, 
               p = int(optimal_n_clusters),
               labels = label_names, 
               leaf_rotation=label_rotation,  # rotates the x axis labels
                leaf_font_size=leaf_font_sizes,  # font size for the x axis labels
               color_threshold = color_threshold
              )
    
    plt.title(f'Hierarchical Clustering Dendrogram ({method} Linkage)')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    return labels,optimal_n_clusters

def plot_3DMDS(sparse_dis_matrix_weighted, labels, gamma=1):
    embedding = MDS(n_components=3, metric=True,
                    eps=1e-5,
                    dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(sparse_dis_matrix_weighted)
    X_arr, Y_arr, Z_arr = zip(*X_transformed)

    cmap = mpl.cm.RdYlBu

#     %matplotlib notebook
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    for (i, xp) in enumerate(X_arr):
        sc = ax.scatter(X_arr[i], Y_arr[i], Z_arr[i], c = 'C0',
                    s = 35)
        ax.text(X_arr[i], Y_arr[i], Z_arr[i], labels[i])

    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    plt.show()

    EL_MDS = euclidean_distances(X_transformed,X_transformed)
    diff = sparse_dis_matrix_weighted- EL_MDS
    n, m = np.shape(diff)
    print(n, m)
    print(np.shape(diff[np.triu_indices(n, k = 1)]))
    stress = np.sum(diff[np.triu_indices(n, k = 1)]**2)
    print(stress)
    norm_stress = np.sqrt(stress /(np.sum(EL_MDS**2)/2))
    print(norm_stress)
    
    return X_transformed

def compute_3DMDS(sparse_dis_matrix_weighted, n_components=3, gamma=1):
    embedding = MDS(n_components=n_components, metric=True,
                    eps=1e-5,
                    dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(sparse_dis_matrix_weighted)
    return X_transformed

def plot_seq(objects_list_shuffled, 
             final_sequence, 
             key, 
             ind_shuffle, 
             grid,
             save_switch=True,
             fname = 'out/OneD_libSeq.pdf'):
    
    data = np.copy(objects_list_shuffled)
    objects_list_ordered = data[final_sequence]
    object_names = key['shortkey']
    shuffled_names = np.array(object_names)[ind_shuffle]
    
    plt.figure(1, figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("input dataset")
    Z = objects_list_shuffled
    plt.pcolormesh(data)
    plt.colorbar()
    length= np.shape(grid)
    length = length[0]
    grid_vis  = grid[0:length:10]
    ind = np.arange(len(grid))[0:length:10]
    grid_vis = np.round(grid_vis, 2)
    plt.xticks(ind, grid_vis , rotation=90)
    r, c = np.shape(objects_list_shuffled)
    plt.yticks(np.arange(r)+0.5, shuffled_names)
    plt.xlabel("Wavelength (mu)")
    plt.ylabel("shuffled index")

    plt.subplot(1, 2, 2)
    plt.title("ordered dataset")
    plt.pcolormesh(objects_list_ordered)
    plt.colorbar()
    plt.xlabel("Wavelength (mu)")

    plt.ylabel("sequenced index")
    length= np.shape(grid)
    length = length[0]
    grid_vis  = grid[0:length:10]
    ind = np.arange(len(grid))[0:length:10]
    grid_vis = np.round(grid_vis, 2)
    plt.xticks(ind, grid_vis , rotation=90)
    r, c = np.shape(objects_list_shuffled)
    plt.yticks(np.arange(r)+0.5, np.array(shuffled_names)[final_sequence])
    plt.tight_layout()
    if save_switch:
        plt.savefig(fname)
    plt.show()
    return