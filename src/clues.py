from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
from sklearn.metrics import euclidean_distances
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy.cluster import hierarchy
import matplotlib.cm as cm
import networkx as nx
from torch_geometric.utils import to_networkx
import gravis as gv
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import silhouette_score

class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgba(self, val):
        return self.scalarMap.to_rgba(val, bytes=True)

    def get_rgb_str(self, val):
        r, g, b, a = self.get_rgba(val)
        return f"rgb({r},{g},{b})"
    
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def set_node_colors(g, cmap='viridis'):
    '''Set colours based on importance values'''

    # scale our colourmap to be between the min-max importance
    vals = []
    for node, data in g.nodes(data=True):
        vals.append(data['clusterid'])
    min_val, max_val = min(vals), max(vals)

    # initialise colour helper
    node_color_generator = MplColorHelper(cmap, min_val, max_val)
    node_colors = {}
    # get rgb string for each node and convert to hex str
    for node, data in g.nodes(data=True):         
        color_rgb = node_color_generator.get_rgb_str(data['clusterid'])
        color_rgb = color_rgb.split('rgb(')[1]
        color_rgb = color_rgb.split(')')
        color_rgb = color_rgb[0].split(',')
        a,b,c = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
        hex_code = rgb_to_hex(a, b, c)
        node_colors[node] = hex_code
    nx.set_node_attributes(g, node_colors, name='color')

def set_node_level_imoprtance(g):
    '''Get node level importance'''
    # sum over each feature importance per node to get the overall node importance
    node_level_importance = np.array(g.graph['node_mask']).sum(axis=1)

    # assign the importance value to each node as an attribute
    node_level_importance_dict = { i : node_level_importance[i] for i in g.nodes }
    nx.set_node_attributes(g, node_level_importance_dict, name="importance")

def set_edge_level_importance(g):
    '''Get edge level importance'''
    edge_level_importance = g.graph['edge_mask']

    # assign the importance value to each edge as an attribute
    edge_level_importance_dict = { edge : edge_level_importance[i] for i, edge in enumerate(g.edges) }
    nx.set_edge_attributes(g, edge_level_importance_dict, name="importance")


def hierarchical_clustering(X, min_clusters, max_clusters, method='single', metric='euclidean', 
                            compute_MDS_switch=True, 
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



def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    return (x,y,z)


def hierarchical_clustering_v2(X, min_clusters, max_clusters, method='single',         
                               metric='euclidean', 
                            compute_MDS_switch=True, label_names = 'bananas', 
                            scatter_label=True, label_rotation = 90, leaf_font_sizes=10, 
                           truncate_mode='level', D_leaf_colors=None,
                            dendrogram_switch = True, 
                               ss_switch = True, 
                               twoD_switch = False, 
                            color_threshold = 0.13, 
                            scatter_size = 5, 
                            scatter_size2 = 5, 
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
    ax = fig.add_subplot(1,2,1, projection='3d')
    
    if twoD_switch:
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_yz = fig.add_subplot(2, 2, 4)
        
    
    if compute_MDS_switch:
        X_mds = compute_3DMDS(X, n_components=3)
#         ax.contour(X, Y, Z, extend3d=True, cmap=cm.coolwarm)
        XX, YY = np.meshgrid(X_mds[:, 0], X_mds[:, 1])
        _, ZZ = np.meshgrid(X_mds[:, 0], X_mds[:, 2])
#         ax.contour(XX, YY, ZZ, c=labels, extend3d=True, cmap='viridis', s= scatter_size)

#         ax.scatter(X_mds[:, 0], X_mds[:, 1], X_mds[:, 2], c=labels, cmap='viridis', s= scatter_size)

        xs_arr = []
        ys_arr = []
        zs_arr = []
# Try Plotting 3D Spheres: 
        cpalette = [val for key, val in colormap_viridis.items()]
        cpalette = cpalette[::-1]
        for i,_ in enumerate(X_mds[:,0]):
            (xs,ys,zs) = drawSphere(X_mds[:, 0][i], X_mds[:, 1][i], X_mds[:, 2][i], r= 0.1/10)
            xs_arr.append(xs)
            ys_arr.append(ys)
            zs_arr.append(zs)
            ax.plot_wireframe(xs, ys, zs,rcount = 50, ccount=50, 
                          color= cpalette[labels[i]-1], alpha = 0.2)
        
        # draw lines
        for i in range(X_mds.shape[0]):
            # z plane
            z_plane = min(X_mds[:, 2])
            z_plane *= 1.1
            ax.plot((X_mds[i, 0], X_mds[i, 0]), (X_mds[i, 1], X_mds[i, 1]), (z_plane, X_mds[i, 2]),
                    color=cpalette[labels[i] - 1], lw=0.2, alpha=0.8)
            
            ax.plot(X_mds[i, 0], X_mds[i, 1], z_plane, 'x',
                    color=cpalette[labels[i] - 1], ms=5, alpha=0.8)
    
        if twoD_switch:
            ax_xy.scatter(X_mds[:, 0], X_mds[:, 1], c= labels, cmap='viridis', s= scatter_size2)
            ax_yz.scatter(X_mds[:, 1], X_mds[:, 2], c= labels, cmap='viridis', s= scatter_size2)
            ax_xy.set_xlabel('X')
            ax_xy.set_ylabel('Y')
            ax_yz.set_xlabel('Y')
            ax_yz.set_ylabel('Z')
        
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
#     ax.set_xlim(-0.1, 0.3)
#     ax.set_ylim(-0.2, 0.2)
#     ax.set_zlim(-0.2, 0.2)
    
    ax.set_aspect('equalxy')
    
    
    # Create an empty list to store silhouette scores
    silhouette_scores = []

    # Perform hierarchical clustering for different number of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        silhouette_scores.append(silhouette_score(X, labels))

    # Find the index of maximum silhouette score
    optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters
    labels = fcluster(Z, t=optimal_n_clusters, criterion='maxclust')
    
    print('Optimal Cluster number is: ', optimal_n_clusters)
    # Plot the number of clusters vs Silhouette score
    if ss_switch:
        ax2 = fig.add_subplot(122)
        ax2.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-', c = 'RoyalBlue')
        ax2.set_title(f'Number of Clusters vs Silhouette Score ({method} Linkage)')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Silhouette Score')
        ax2.axvline(optimal_n_clusters, color='r', linestyle='--', label='Optimal Number of Clusters')
        ax2.legend()
    
# Visualize the dendrogram
#     cpalette = ['#fde725','#5ec962','#21918c','#3b528b','#440154']
#     cpalette = [val for key, val in colormap_viridis.items()]
    hierarchy.set_link_color_palette(cpalette)
    print(int(optimal_n_clusters))
    
    if dendrogram_switch: 
        fig = plt.figure(figsize=(12, 7))
        
        cluster_colors = cpalette
        cluster_colors_array = [cluster_colors[l - 1] for l in labels]
        print(cluster_colors_array)
        
        link_cols = {}
        for i, i12 in enumerate(Z[:,:2].astype(int)):
            c1, c2 = (link_cols[x] if x > len(Z) else cluster_colors_array[x] for x in i12)
            link_cols[i+1+len(Z)] = c1 if c1 == c2 else 'gray'
        
        
        d = dendrogram(Z, 
            # p = int(optimal_n_clusters),
            labels = label_names, 
            leaf_rotation=label_rotation,  # rotates the x axis labels
            leaf_font_size=leaf_font_sizes,  # font size for the x axis labels
            color_threshold = None,
            link_color_func = lambda x: link_cols[x]
        )
        
        
        leaves = hierarchy.leaves_list(Z)
        print(leaves)
        for i, leaf in enumerate(plt.gca().get_xticklabels()):
            leaf.set_color(cpalette[labels[leaves[i]] - 1])
    
        plt.title(f'Hierarchical Clustering Dendrogram ({method} Linkage)')
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
    return labels,optimal_n_clusters