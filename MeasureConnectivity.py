import networkx as nx
#import community.community_louvain as community
import numpy as np
import matplotlib.pyplot as plt
from random import randint,random
import community as community_louvain
#from networkx.algorithms.community import greedy_modularity_communities
import networkx as nx
import matplotlib.pyplot as plt

def ConvertToAdjaency(A):
    G=nx.Graph()
    for i in range(N):
        for j in range(N):
            if A[i,j]>0:
                G.add_edge(i,j,weight=A[i,j])
    partition=community.best_partition(G)
    pos = nx.spring_layout(G)  # compute graph layout
    plt.figure(figsize=(8, 8))  # image is 8 x 8 inches
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=600, cmap=plt.cm.RdYlGn, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    plt.show(G)

def GrowConnectivity(G):
    algebraic_connectivity=[]
    for k in range(2):
        algebraic_connectivity.append([])
        for i in range(N):
            G.add_edge(randint(0,N-1),randint(0,N-1))
            algebraic_connectivity[k].append(nx.linalg.algebraic_connectivity(G))


