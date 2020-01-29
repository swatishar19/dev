from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import networkx as nx
# from context import PROJECT_DIR

fpxaid = pd.read_csv('aidXfpP-valuebinary.csv', index_col=0)

# print(fpxaid.head(50))
X = pairwise_distances(fpxaid, metric='jaccard')
X = pd.DataFrame(X, index=fpxaid.index, columns=fpxaid.index)
# print(X.head())

# X.to_csv('aidXtoxprintfp_JaccardDistances.csv')

# write to a graphml file
G = nx.from_numpy_matrix(X.values)
# print(G.nodes)

# nodes are just indices, so relabel them as AIDs
mapping = dict(list(zip(G.nodes(), X.index.tolist())))
nx.relabel_nodes(G, mapping, copy=False)

print("There are {0} nodes and {1} edges before filtering".format(len(G.nodes()), len(G.edges())))

# remove edges whose edge is > 0.75
edges_to_remove = [edge for edge in G.edges(data=True)
                   if edge[2]['weight'] > 0.75]
G.remove_edges_from(edges_to_remove)

# now remove nodes whose degree is 0

nodes_to_remove = [node for node in G.nodes() if G.degree(node) == 0]
G.remove_nodes_from(nodes_to_remove)

print("There are {0} nodes and {1} edges after filtering".format(len(G.nodes()), len(G.edges())))
# write to file for later use

nx.write_graphml(G, 'AID_graph2.graphml')
