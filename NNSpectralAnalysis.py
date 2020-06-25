#!/usr/bin/env python
# coding: utf-8
# %%
import networkx as nx
from community import community_louvain
import numpy as np
import matplotlib.pyplot as plt
from random import randint,random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import scipy.sparse as sparse
#from visualization import RANDOM_STATE
#from spectral_cluster_model import cluster_net,build_cluster_graph,delete_isolated_ccs,weights_array_to_cluster_quality
#from Results.NNSpectralAnalysis import SOMVisualization,WeightsToAdjaency,GraphPartition
from collections import Counter
from matplotlib.ticker import MaxNLocator
from minisom import MiniSom
import scipy.sparse as sparse


# %%
"""
dataset="Cora"
file_constraited=dataset+"Convergence"

for epoch in range(0,200,40):
    weights_path0=dataset+"Convergence/WeightChanges-Cora-GCN-param_512_2_0.99_0.2-monte_0-"+str(epoch)+".pt"
    weights_array0=torch.load(weights_path0)
    weights_array=[]
    for (i, weight) in enumerate(weights_array0):
        weight = weight.cpu().detach().numpy()
        weights_array.append(weight)
    weights_path=dataset+"Convergence/WeightChanges-Cora-GCN-param_512_2_0.99_0.2-monte_0-"+str(epoch)+".pckl"  
    pickle.dump(weights_array,open(weights_path,'wb'))
    
    for weights in weights_array:
        G=WeightsToAdjaency(weights)

        num_clusters=6
        assign_labels = 'kmeans'
        eigen_solver = 'amg'
        epsilon=1e-8
        #[ncut_val, clustering_labels]=weights_array_to_cluster_quality(weights_array, adj_mat, num_clusters,eigen_solver, assign_labels, epsilon,is_testing=False)
        #    weights_array_to_cluster_quality() 
        #[labels, metrics]=run_spectral_cluster(weights_path)
        plot_eigenvalues(weights_array)
        labels=cluster_net(num_clusters, adj_mat, eigen_solver, assign_labels)
        build_cluster_graph(weights_path,labels,normalize_in_out=True)"""


# %%
def ToBlockMatrix(weights):
    M,N=weights.shape
    A11=np.zeros((M,M))
    A12=weights
    A21=np.transpose(weights)
    A22=np.zeros((N,N))
    BlockMatrix = np.block([[A11, A12], [A21, A22]])
    return BlockMatrix



def WeightsToAdjaency(Weights):
    M,N=Weights.shape
    GWeight=nx.Graph()
    GWeight.add_nodes_from(range(M+N))
    G1=GWeight
    for i in range(M):
        for j in range(N):
            GWeight.add_weighted_edges_from([(i,j+M,Weights[i,j])])
            G1.add_edge(i,j+M)
            
    """print("Diconnected points is {}".format(list(nx.isolates(G))))
    G.remove_nodes_from(list(nx.isolates(G)))"""
    return GWeight,G1


def GraphPartition(G):
    G.remove_nodes_from(list(nx.isolates(G)))
    #Degree_distribution(G)
    partition=community_louvain.best_partition(G)
    pos=community_layout(G,partition)
    return pos,partition


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos

def _position_communities(g, partition, **kwargs):

    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos

def _find_between_community_edges(g, partition):

    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges

def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos

def test():
    # to install networkx 2.0 compatible version of python-louvain use:
    # pip install -U git+https://github.com/taynaud/python-louvain.git@networkx2
    from community import community_louvain

    g = nx.karate_club_graph()
    partition = community_louvain.best_partition(g)
    pos = community_layout(g, partition)
    nx.draw(g, pos, node_color=list(partition.values())); plt.show()
    return




# %%
def Degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
    degreeCount=Counter(degree_sequence)
    deg,cnt=zip(*degreeCount.items())
    fig =plt.subplot()
    plt.bar(deg,cnt,width=0.08,color='b')
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()
    
def SOMVisualization(data,som_shape,N):
    som = MiniSom(som_shape[0], som_shape[1], N, sigma=.5, learning_rate=.5,
                  neighborhood_function='gaussian', random_seed=10)

    #som.pca_weights_init(data)
    som.train_batch(data, 500, verbose=True)


    # each neuron represents a cluster
    winner_coordinates = np.array([som.winner(x) for x in data]).T
    # with np.ravel_multi_index we convert the bidimensional
    # coordinates to a monodimensional index
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
    # plotting the clusters using the first 2 dimentions of the data
    for c in np.unique(cluster_index):
        plt.scatter(data[cluster_index == c, 0],
                    data[cluster_index == c,1], label='cluster='+str(c), alpha=.7)

    # plotting centroids
    for centroid in som.get_weights():
        plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                    s=80, linewidths=35, color='k', label='centroid')
    plt.legend()
    plt.show()
    
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    n, d = A.shape

    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)

    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

if __name__=="__main__":
    dataset="Cora"
    file_constraited=dataset+"Convergence"
    from minisom import MiniSom    

    for epoch in range(200,220,40):
        weights_path=dataset+"Convergence/WeightChanges-Cora-GCN-param_512_2_0.99_0.2-monte_0-"+str(epoch)+".pt"
        print(weights_path)
        weights_array=torch.load(weights_path)
        for weights in weights_array:
            weights=weights.cpu().detach().numpy()
            adj_mat=WeightsToAdjaency(weights)
            num_clusters=6
            assign_labels = 'kmeans'
            eigen_solver = 'amg'
            epsilon=1e-8
            """labels=cluster_net(num_clusters, adj_mat, eigen_solver, assign_labels)
            build_cluster_graph(adj_mat,labels,normalize_in_out=True)"""
            G=WeightsToAdjaency(weights)
            GraphPartition(G)
        """
            N=data.shape[1]
            som_shape=(6,6)
            SOMVisualization(data,som_shape,N)"""
            #Adjaencypartition(data)


# %%
