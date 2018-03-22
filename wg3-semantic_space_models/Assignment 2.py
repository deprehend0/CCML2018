#%% Importing all libraries
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cosine

from treelib import Node, Tree

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
#%% Importing all datasets
# Google File
model = KeyedVectors.load_word2vec_format('./glove.6B.50d.txt.word2vec', binary=False)
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

# Glove file
#glove_input_file = 'glove.6B.50d.txt'
#word2vec_output_file = 'glove.6B.100d.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)
#model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
#%% Parameters
# Clusters is 2 (Default)
# Clusters is 5 (Head, Torso, Leg, Arm, Hand)
# Clusters is 6 (Head, Face, Torso, Leg, Arm, Hand)
# Clusters is 7 (Head, Face, Mouth, Torso, Leg, Arm, Hand)
n_clusters = 2
#%% Functions
def compute_paths_distance(tree, sort=True):
    # Create the graph for the tree in order to compute all paths' length
    graph = nx.Graph()
    graph.add_nodes_from(tree.nodes.keys())
    for p in tree.paths_to_leaves():
        graph.add_path(p)
    paths = nx.all_pairs_shortest_path(graph)
    distance = [(n1,n2,path,len(path)-1) for n1,nodes in paths.iteritems() 
                                         for n2,path  in nodes.iteritems()]
    # Penalty function lower leaves will get higher values, while higher leaves will get a penalty
    depth = tree.depth()
    level_malus = [100*(depth-lvl)/depth for lvl in range(depth+1)]
    for i,(n1,n2,path,dist) in enumerate(distance):
        highest_level = min(tree.level(n) for n in path)
        if tree.parent(n1) == tree.parent(n2):       dist = 1
        # Rescale distance according to depth
        # To make distance more manageable it will be multiplied by 100, in case of 0.001 and/or lower
        if dist > 0:
            scaled_dist = level_malus[highest_level] + dist*100
            distance[i] = (n1,n2,path,scaled_dist)
    if sort:
        return sorted(distance), nx.adjacency_matrix(graph)
    return distance, nx.adjacency_matrix(graph)

def compute_distance_matrix(distance, n):
    distance_matrix = np.zeros(shape=(n,n), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = distance[i*n + j][3]        
    return distance_matrix / np.max(distance_matrix)

def compute_clusters(distance_matrix, names, n_clusters=n_clusters, connectivity_matrix=None, linkage='complete'):
    cluster = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage=linkage, connectivity=connectivity_matrix)
    cluster.fit(distance_matrix)  
    return cluster 

def plot_distance_matrix(distance_matrix, labels):
    n = distance_matrix.shape[0]    
    plt.imshow(distance_matrix, cmap='hot')
    plt.xticks(range(n), labels, rotation=90, fontsize=7)
    plt.yticks(range(n), labels, fontsize=7)
    plt.colorbar()
    plt.show()
    
def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    plt.xticks(rotation=90)
    plt.show()
    
def get_w2v_cluster(model, names):
    # Compute the similarity matrix according to word2vec
    distance_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = model.distance(body_parts[i], body_parts[j])
    plot_distance_matrix(distance_matrix, body_parts)
    w2v = compute_clusters(distance_matrix, body_parts, 5)
    return w2v
#%% Creating the tree
ref_hierarchy = Tree()
# Level 0
# Creating the ancestor node for all eventual nodes
ref_hierarchy.create_node('Body', 'body')
# Level 1
# Creating the four main clusters: Head, Trunk/Torso, Arm, Leg
ref_hierarchy.create_node('Head', 'head', parent='body')
ref_hierarchy.create_node('Torso', 'torso', parent='body')
ref_hierarchy.create_node('Arm', 'arm', parent='body')
ref_hierarchy.create_node('Leg', 'leg', parent='body')
# Level 2
# Creating the Face node with the Mouth child
ref_hierarchy.create_node('Face', 'face', parent='head')
ref_hierarchy.create_node('Mouth', 'mouth', parent='face')
ref_hierarchy.create_node('Nose', 'nose', parent='face')
ref_hierarchy.create_node('Lips', 'lips', parent='face')
ref_hierarchy.create_node('Eyes', 'eyes', parent='face')
ref_hierarchy.create_node('Iris', 'iris', parent='eyes')
ref_hierarchy.create_node('Pupil', 'pupil', parent='eyes')
ref_hierarchy.create_node('Ears', 'ears', parent='face')
ref_hierarchy.create_node('Eyebrows', 'eyebrows', parent='face')
ref_hierarchy.create_node('Cheecks', 'cheeks', parent='face')
ref_hierarchy.create_node('Chin', 'chin', parent='face')
ref_hierarchy.create_node('Forehead', 'forehead', parent='face')
ref_hierarchy.create_node('Teeth', 'teeth', parent='mouth')
ref_hierarchy.create_node('Tongue', 'tongue', parent='mouth')
# Level 3
# Creating the Arm node with the Fingers child
ref_hierarchy.create_node('Bicep', 'bicep', parent='arm')
ref_hierarchy.create_node('Elbow', 'elbow', parent='arm')
ref_hierarchy.create_node('Forearm', 'forearm', parent='arm')
ref_hierarchy.create_node('Wrist', 'wrist', parent='arm')
ref_hierarchy.create_node('Hand', 'hand', parent='arm')
# Hands and fingers
ref_hierarchy.create_node('Palm', 'palm', parent='hand')
ref_hierarchy.create_node('Fingers', 'fingers', parent='hand')  
ref_hierarchy.create_node('Pinky', 'pinky', parent='fingers')
ref_hierarchy.create_node('Thumb', 'thumb', parent='fingers')

# Level 4
# Creating the Leg node
ref_hierarchy.create_node('Thigh', 'thigh', parent='leg')
ref_hierarchy.create_node('Knee', 'knee', parent='leg')
ref_hierarchy.create_node('Calf', 'calf', parent='leg')
ref_hierarchy.create_node('Ankle', 'ankle', parent='leg')
ref_hierarchy.create_node('Foot', 'foot', parent='leg')
# Level 5
# Creating the foot node
ref_hierarchy.create_node('Toes', 'toes', parent='foot')
# level 6
# Creating the torso node
ref_hierarchy.create_node('Neck', 'neck', parent='torso')
ref_hierarchy.create_node('Shoulders', 'shoulder', parent='torso')
ref_hierarchy.create_node('Belly', 'belly', parent='torso')
ref_hierarchy.create_node('Back', 'back', parent='torso')
ref_hierarchy.create_node('Hips', 'hips', parent='torso')
ref_hierarchy.show()

#%% Applying functions
body_parts = sorted(ref_hierarchy.nodes.keys())
distances, connectivity_matrix = compute_paths_distance(ref_hierarchy)
distance_matrix = compute_distance_matrix(distances,len(body_parts))
w2v = get_w2v_cluster(model, body_parts)
n = len(body_parts)

# Reference 
print('Reference')
cl_ref = compute_clusters(distance_matrix, body_parts, n_clusters)
plot_dendrogram(cl_ref, labels=body_parts)

# Compute the similarity matrix according to word2vec
print('\nword2vec')

distance_matrix_w2v = np.zeros(shape=(n,n), dtype='float64')
for i,j in np.ndindex((n,n)):
    distance_matrix_w2v[i,j] = model.distance(body_parts[i], body_parts[j])
    
plot_distance_matrix(distance_matrix_w2v, body_parts)
w2v = compute_clusters(distance_matrix_w2v, body_parts, n_clusters)
plot_dendrogram(w2v, labels=body_parts)

print('word2vec vs REF: {}'.format(adjusted_rand_score(cl_ref.labels_, w2v.labels_)))
