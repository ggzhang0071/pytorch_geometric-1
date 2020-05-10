import networkx as nx
import community
import numpy as np
import matplotlib.pyplot as plt
from random import randint,random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import pickle
import scipy.sparse as sparse
from visualization import RANDOM_STATE
from spectral_cluster_model import clustering_experiment, weights_to_graph,cluster_net,delete_isolated_ccs,weights_array_to_cluster_quality
from visualization import run_spectral_cluster,build_cluster_graph,plot_eigenvalues


dataset="Cora"
file_constraited=dataset+"Convergence"

for epoch in range(0,200,40):
    weights_path0=dataset+"Convergence/WeightChanges-Cora-GCN-param_512_2_0.99_0.2-monte_0-"+str(epoch)+".pt"
    weights_array0=torch.load(weights_path0)
    weights_array=[]
    for (i, weight) in enumerate(weights_array0):
        weight = weight[:800,:800].cpu().detach().numpy()
        weights_array.append(weight)
    weights_path=dataset+"Convergence/WeightChanges-Cora-GCN-param_512_2_0.99_0.2-monte_0-"+str(epoch)+".pckl"  
    pickle.dump(weights_array,open(weights_path,'wb'))
    
    
    adj_mat=weights_to_graph(weights_array)
    new_weight_array, new_adj_mat=delete_isolated_ccs(weights_array, adj_mat)
   
    num_clusters=6
    assign_labels = 'kmeans'
    eigen_solver = 'amg'
    epsilon=1e-8
    #[ncut_val, clustering_labels]=weights_array_to_cluster_quality(weights_array, adj_mat, num_clusters,eigen_solver, assign_labels, epsilon,is_testing=False)
    #    weights_array_to_cluster_quality() 
    #[labels, metrics]=run_spectral_cluster(weights_path)
    plot_eigenvalues(weights_path)
    #labels=cluster_net(num_clusters, adj_mat, eigen_solver, assign_labels)
    #build_cluster_graph(weights_path,labels,normalize_in_out=True)




