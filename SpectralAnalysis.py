import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import randint,random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle,os,math
import scipy.sparse as sparse
from collections import Counter
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import DBSCAN,SpectralClustering
import cupy as cp
from torch.utils.dlpack import from_dlpack, to_dlpack
from cupy.core.dlpack import toDlpack,fromDlpack
import link_prediction as lp


def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]
    
    
def weights_array_to_cluster_quality(weights_array, adj_mat, num_clusters,
                                     eigen_solver, assign_labels, epsilon,
                                     is_testing=False):
    # t1 = time.time()
    clustering_labels = cluster_net(num_clusters, adj_mat, eigen_solver, assign_labels)
    # t2 = time.time()
    ncut_val = compute_ncut(adj_mat, clustering_labels, epsilon, verbose=is_testing)

    if is_testing:
        ncut_val_previous_method = ncut(weights_array, num_clusters, clustering_labels, epsilon)
        print('NCUT Current', ncut_val)
        print('NCUT Previous', ncut_val_previous_method)
        assert math.isclose(ncut_val, ncut_val_previous_method, abs_tol=1e-5)

    return ncut_val, clustering_labels


def WeightsToAdjaency(Weights,startNodeNums):
    M,N=Weights.shape
    GWeight=nx.Graph()
    GWeight.add_nodes_from(range(M+N))
    G1=GWeight
    for i in range(M):
        for j in range(N):
            GWeight.add_weighted_edges_from([(i+startNodeNums,j+M+startNodeNums,Weights[i,j])])

            G1.add_edge(i,j+M)
            
    """print("Diconnected points is {}".format(list(nx.isolates(G))))
    G.remove_nodes_from(list(nx.isolates(G)))"""
    GWeight=GWeight.to_undirected()
    G1=G1.to_undirected()
    return GWeight,G1


def GraphPartition(G):
    G.remove_nodes_from(list(nx.isolates(G)))
    #Degree_distribution(G)
    partition=community_louvain.best_partition(G)
    pos=community_layout(G,partition)
    return pos,partition

def cluster_net(n_clusters, adj_mat, eigen_solver, assign_labels):
    cluster_alg = SpectralClustering(n_clusters=n_clusters,
                                     eigen_solver=eigen_solver,
                                     affinity='precomputed',
                                     assign_labels=assign_labels)
    clustering = cluster_alg.fit(adj_mat)
    return clustering.labels_

def delete_isolated_ccs(weight_array, adj_mat):
    # find connected components that aren't represented on both the first and
    # the last layer, and delete them from the graph

    nc, labels = sparse.csgraph.connected_components(adj_mat, directed=False)

    # if there's only one connected component, don't bother
    if nc == 1:
        return weight_array, adj_mat
    
    widths = weights_to_layer_widths(weight_array)

    # find cc labels represented in the first layer
    initial_ccs = set()
    for i in range(widths[0]):
        initial_ccs.add(labels[i])
    # find cc labels represented in the final layer
    final_ccs = set()
    final_layer = len(widths) - 1
    for i in range(widths[-1]):
        neuron = mlp_tup_to_int((final_layer, i), widths)
        final_ccs.add(labels[neuron])

    # find cc labels that aren't in either of those two sets
    isolated_ccs = set()
    for c in range(nc):
        if not (c in initial_ccs and c in final_ccs):
            isolated_ccs.add(c)

    # if there aren't any isolated ccs, don't bother deleting them!
    if not isolated_ccs:
        return weight_array, adj_mat

    # go through weight_array
    # for each array, go to the rows and cols
    # figure out which things you have to delete, then delete them
    new_weight_array = []
    for (t, mat) in enumerate(weight_array):
        # print("weight array number:", t)
        n_rows, n_cols = mat.shape
        # print("original n_rows, n_cols:", (n_rows, n_cols))
        rows_layer = t
        cols_layer = t + 1

        # delete rows and cols corresponding to neurons in isolated clusters
        rows_to_delete = []
        for i in range(n_rows):
            neuron = mlp_tup_to_int((rows_layer, i), widths)
            if labels[neuron] in isolated_ccs:
                rows_to_delete.append(i)

        cols_to_delete = []
        for j in range(n_cols):
            neuron = mlp_tup_to_int((cols_layer, j), widths)
            if labels[neuron] in isolated_ccs:
                cols_to_delete.append(j)

        # print("rows to delete:", rows_to_delete)
        # print("columns to delete:", cols_to_delete)

        rows_deleted = np.delete(mat, rows_to_delete, 0)
        new_mat = np.delete(rows_deleted, cols_to_delete, 1)
        # print("new mat shape:", new_mat.shape)
        new_weight_array.append(new_mat)

    # then return the adj_mat
    new_adj_mat = weights_to_graph(new_weight_array)
    return new_weight_array, new_adj_mat



def weights_to_graph(weights_array):
    # take an array of weight matrices, and return the adjacency matrix of the
    # neural network it defines.
    # if the weight matrices are A, B, C, and D, the adjacency matrix should be
    # [[0   A^T 0   0   0  ]
    #  [A   0   B^T 0   0  ]
    #  [0   B   0   C^T 0  ]
    #  [0   0   C   0   D^T]
    #  [0   0   0   D   0  ]]
    # however, the weight matrices we get are A^T etc.

    block_mat = []

    # for everything in the weights array, add a row to block_mat of the form
    # [None, None, ..., sparsify(np.abs(mat)), None, ..., None]
    for (i, mat) in enumerate(weights_array):
        mat=np.maximum(mat,0)         
        sp_mat = sparse.coo_matrix(np.abs(mat))
        if i == 0:
            # add a zero matrix of the right size to the start of the first row
            # so that our final matrix is of the right size
            n = mat.shape[0]
            first_zeroes = sparse.coo_matrix((n, n))
            block_row = [first_zeroes] + [None]*len(weights_array)
        else:
            block_row = [None]*(len(weights_array)+1)
        block_row[i+1] = sp_mat
        block_mat.append(block_row)

    # add a final row to block_mat that's just a bunch of [None]s followed by a
    # zero matrix of the right size
    m = weights_array[-1].shape[1]
    final_zeroes = sparse.coo_matrix((m, m))
    nones_row = [None]*len(weights_array)
    nones_row.append(final_zeroes)
    block_mat.append(nones_row)

    # turn block_mat into a sparse matrix
    up_tri = sparse.bmat(block_mat, 'csr')

    # we now have a matrix that looks like
    # [[0   A^T 0   0  ]
    #  [0   0   B^T 0  ]
    #  [0   0   0   C^T]
    
    #  [0   0   0   0  ]]
    # add this to its transpose to get what we want
    adj_mat = up_tri + up_tri.transpose()
    return adj_mat

def ToBlockMatrix(weights):
    M,N=weights.shape
    A11=np.zeros((M,M))
    A12=weights
    A21=np.transpose(weights)
    A22=np.zeros((N,N))
    BlockMatrix = np.block([[A11, A12], [A21, A22]])
    return BlockMatrix


def Compute_fiedler_vector(G):
    nrom_laplacian_matrics = nx.normalized_laplacian_matrix(G,weight='weight')
    nrom_laplacian_matrics_cupy=cp.asarray(nrom_laplacian_matrics.toarray())
    w,v=cp.linalg.eigh(nrom_laplacian_matrics_cupy)
    #algebraic_connectivity,fiedler_vector=power_iteration(nrom_laplacian_matrics.)
    algebraic_connectivity = w[1] # Neat measure of how tight the graph is
    fiedler_vector = v[:,1].T
    fiedler_vector=torch.Tensor(cp.asarray(cp.real(fiedler_vector)))
    algebraic_connectivity=torch.Tensor(cp.asarray(algebraic_connectivity))
    return algebraic_connectivity, fiedler_vector

def Fiedler_vector_cluster(G,startClassi):
    
    algebraic_connectivity, fiedler_vector=Compute_fiedler_vector(G)
    PartOne=[]
    PartTwo=[]
    for node in range(G.number_of_nodes()):
        if fiedler_vector[node]<0:
            PartOne.append(node)
        else:
            PartTwo.append(node)
    PartitionResults={str(startClassi+0):PartOne,str(startClassi+1):PartTwo}
    G1=nx.subgraph(G,PartOne)
    G2=nx.subgraph(G,PartTwo)
    return G1,G2,PartitionResults

def chooseSemiMatrix(Weight,locx,M):
    semipart=int((M-1)/2)
    if locx<semipart:
        ChooseWeights=Weight[:M]
    elif locx>=semipart and locx<(M-semipart):
            ChooseWeights=Weight[locx-semipart:locx+semipart+1]
    else:
            ChooseWeights=Weight[-M:]
    return ChooseWeightspp 


def WeightedLinkPrediction(G,cluters,LinkPredictionMethod,VectorPairs):
    PartitionClassi=set([*cluters.keys()])
    predLinkWeight=[]
    AddLinkGraph=nx.Graph()
    for OneClassi in PartitionClassi:
        oneClassNodes=cluters[OneClassi]
        SubGraph=nx.Graph()
        SubGraph.add_nodes_from(oneClassNodes)
        #l
        for (i,j) in G.edges:
            if (i in SubGraph.nodes()) and (j in SubGraph.nodes()) and 'weight' in G.get_edge_data(i,j):
                    SubGraph.add_weighted_edges_from([(i,j,G.get_edge_data(i,j)['weight'])])
            else:
                continue
        if SubGraph.number_of_edges()>=2:
            diag,vector=Compute_fiedler_vector(SubGraph)
            for iter1 in range(VectorPairs):
                if torch.min(vector)<0:
                    locx=torch.argmax(vector).tolist()
                    locy=torch.argmin(vector).tolist()
                    StartNode=oneClassNodes[locx]
                    EndNode=oneClassNodes[locy]
                    WrongLink=[(StartNode,EndNode)]
                    vector=np.delete(vector,locx)
                    vector=np.delete(vector,locy)
                    #AddLinkGraph.add_edge(StartNode,EndNode)
                    preds=getattr(nx,LinkPredictionMethod)(SubGraph,WrongLink)
                    for u,v,p in preds:
                        predLinkWeight.append((u,v,p))
        else:
            continue
       
            
        ## save
        """AddedLinkGraphResultsFiles= "Results/PartitionResults/AddedLindedGraph.pkl"
        fwG=open(AddedLinkGraphResultsFiles,'wb')
        pickle.dump(G,fwG)"""
    return predLinkWeight


def WeightCorrection(classiResultsFiles,num_classes,GraphResultsFiles,OptimizedNet,PredAddEdgeResults,LinkPredictionMethod,VectorPairs,UseOld):
    if os.path.exists(PredAddEdgeResults) and UseOld==True:
        predLinkWeight=np.load(PredAddEdgeResults)
    else:
        Graph_array,Gragh_unwighted_array=[],[]
        LayerNodeNum=[]
        startNodeNums=0
        state_dict = OptimizedNet.state_dict()
        if os.path.exists(classiResultsFiles) and os.path.exists(GraphResultsFiles) and UseOld==True:
            frC=open(classiResultsFiles,'rb')
            PartitionResults=pickle.load(frC)

            frG=open(GraphResultsFiles,'rb')
            G=pickle.load(frG)
            L=nx.adjacency_matrix(G)
            incidence_matrix=nx.incidence_matrix(G)
            algebraic_connectivity,fiedler_vector=Compute_fiedler_vector(G)

        else:
            for layer_name in state_dict:
                if ("layers" in layer_name) and ("weight" in layer_name):
                    Weight=state_dict[layer_name]
                    print(Weight.shape)
                    if Weight.dim()==3:
                        Weight=torch.squeeze(Weight)
                        DimCompress=True
                    Weight=Weight.cpu().detach().numpy()
                    Gone,G_unweighted=WeightsToAdjaency(Weight,startNodeNums)
                    startNodeNums+=Gone.number_of_nodes()
                    LayerNodeNum.append(Gone.number_of_nodes())
                    Graph_array.append(Gone)
                    Gragh_unwighted_array.append(G_unweighted)
            G= nx.compose(Graph_array[0],Graph_array[1])
            Gu= nx.compose(Gragh_unwighted_array[0],Gragh_unwighted_array[1])
            L=nx.adjacency_matrix(G)
            incidence_matrix=nx.incidence_matrix(Gu)
            #comps=nx.connected_components(G)
            G_array=[G]
            iter1=0
            while len(G_array) < num_classes and iter1<math.log(num_classes,2)+1:
                G_array_tmp=[]
                PartitionResults={}
                for i in range(len(G_array)):
                    if G_array[i].number_of_edges()>0:
                        G1,G2,PartitionResults1=Fiedler_vector_cluster(G_array[i],0+2*i)
                        G_array_tmp.append(G1)
                        G_array_tmp.append(G2)
                        PartitionResults.update(PartitionResults1)
                    else:
                        continue
                iter1+=1
                G_array=G_array_tmp

            ### saving
            fwC=open(classiResultsFiles,'wb')
            pickle.dump(PartitionResults,fwC)

            fwG=open(GraphResultsFiles,'wb')
            pickle.dump(G,fwG)
        predLinkWeight=WeightedLinkPrediction(G,PartitionResults,LinkPredictionMethod,VectorPairs)
        np.save(PredAddEdgeResults,predLinkWeight)

    if len(predLinkWeight)==0:
        pass
    else:
        print(predLinkWeight)
        state_dict = OptimizedNet.state_dict()
        NeededAddEdges=[]
        for layer_name in state_dict:
            if ("layers" in layer_name) and ("weight" in layer_name):
                Weight=state_dict[layer_name]
                if Weight.dim()==3:
                    Weight=torch.squeeze(Weight)
                    DimCompress=True
                else:
                    DimCompress=False
                M,N=Weight.shape
                BaseNode=0
                for iter1 in range(len(predLinkWeight)):
                    if BaseNode<=predLinkWeight[iter1][0]<(BaseNode+M) and (BaseNode+M)<=predLinkWeight[iter1][1]<=(BaseNode+M+N):
                        Weight[int(predLinkWeight[iter1][0]-BaseNode),int(predLinkWeight[iter1][1]-M-BaseNode)]=predLinkWeight[iter1][2]
                        print("Weight change from {} to {} at connection between node {} to {} weight errors.".format(round(Weight[int(predLinkWeight[iter1][0]-BaseNode),int(predLinkWeight[iter1][1]-M-BaseNode)],4),round(predLinkWeight[iter1][2],4) ,int(predLinkWeight[iter1][0]-BaseNode),int(predLinkWeight[iter1][1]-M-BaseNode)))
        
                    elif ((BaseNode<=predLinkWeight[iter1][0]<BaseNode+M and BaseNode<predLinkWeight[iter1][1]<=(BaseNode+M))) or ((BaseNode+M<=predLinkWeight[iter1][0]<(BaseNode+M+N) and (BaseNode+M)<predLinkWeight[iter1][1]<=(BaseNode+M+N))):
                        print("Need add peer topology from node {} to {}".format(int(predLinkWeight[iter1][0]-BaseNode),int(predLinkWeight[iter1][1]-BaseNode)))
                        NeededAddEdges.append(predLinkWeight[iter1])
                    else:
                        print("Topology wrong that need correct from node {} to {}".format(int(predLinkWeight[iter1][0]-BaseNode),int(predLinkWeight[iter1][1]-BaseNode)))
                        
                if DimCompress==True:
                    state_dict[layer_name]=torch.unsqueeze(Weight,0)
                else:
                    pass
                BaseNode=M+N

        OptimizedNet.load_state_dict(state_dict)

    return OptimizedNet
        
        
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

def power_iteration(A):
    n, d = A.shape
    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    np.seterr(divide='ignore',invalid='ignore')
    while True:
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)

        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) < 0.01:
            break

        v = v_new
        ev = ev_new

    return ev_new, v_new

def UpdateWeights(net,Parition_array):
    #state_dict=torch.load("Net_state_dict")
    state_dict = net.state_dict()
    for i,name in enumerate(state_dict):
        Weight=state_dict[weight]
        Weight=CorrectWeights(Weight,Parition_array[i])
        state_dict[name]=Weight
        net.load_state_dict(state_dict) 
    return net             

def Adjaencypartition(BlockMatrix):
    BlockMatrix=np.maximum(BlockMatrix,0)          
    sp_mat = sparse.coo_matrix(BlockMatrix)
    G=nx.from_scipy_sparse_matrix(sp_mat)
    G.remove_nodes_from(list(nx.isolates(G)))
    partition=community_louvain.best_partition(G)
    """pos=community_layout(G,partition)
    nx.draw(G, pos, node_color=list(partition.values()))
    plt.show()"""
    #Degree_distribution(G)
    return partition

def Degree_distribution(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True) 
    degreeCount=Counter(degree_sequence)
    deg,cnt=zip(*degreeCount.items())
    fig =plt.subplot()
    plt.bar(deg,cnt,width=0.08,color='b')
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.show()

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
