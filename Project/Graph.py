import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_excel('../data/User_Info.xlsx')

#Vector size 7233
vector = df['User Info']
num_connections = df['num-connections']

stopwords_list = stopwords.words('dutch') + stopwords.words('english') + stopwords.words('french')
vectorizer = TfidfVectorizer(analyzer='word', stop_words=stopwords_list)

tfidf_matrix = vectorizer.fit_transform(vector)
print(vector.shape)
#TFIDF Matrix - Vector size for comparison with the entire matrix
tfidf_matrix = tfidf_matrix[0:7233]
sq_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
sq_matrix[sq_matrix >= 0.5] = 1
sq_matrix[sq_matrix < 0.5] = 0

sum_ = np.sum(sq_matrix,axis=1).tolist()

# X is the Case Study Recruiter
# Y is the Recommended Person
# Z max 2.0, indirect path, max 1.0 no direct path
# Check X if Recommended Person index got a 1 or 0 in the array
X = np.array(sq_matrix[1851])
Y = np.array(sq_matrix[454])
Z = X + Y

rows, cols = np.where(sq_matrix == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr, node_size=50, with_labels=True)
plt.show()

poi = 18 # person of interest, index number
relevant_persons = nx.ego_graph(gr, poi, center=True, undirected=False, distance=None)

# To get the other (i.e., out of range people) you calculate the subgraph within range with the code above.
# Then you give that graph with the tfidf_matrix to the code below and you get the persons that are out of range.
def get_out_of_range(in_range_graph, matrix):
    in_range_nodes = in_range_graph.nodes()
    tmp_array = []
    for i in range(len(matrix)):
        if i in in_range_nodes:
            tmp_array.append(matrix[i])

    return tmp_array

