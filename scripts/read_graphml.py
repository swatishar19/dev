import networkx as nx
import pandas as pd

G = nx.read_graphml('AIDs22clusters.graphml')

df = pd.DataFrame([data for label, data in G.nodes(data=True)])

df.to_csv('AIDs22clusters.csv')
# print(df.head())