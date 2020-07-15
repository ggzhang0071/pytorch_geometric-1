import networkx as nx
import matplotlib.pyplot as plt
from SpectralAnalysis import community_layout
import pickle
dataset="Pubmed"
modelName="GCN"
for epoch in range(40,80,20):
    classiResultsFiles="Results/PartitionResults/{}-{}-oneClassNodeEpoch_{}.pkl".format(dataset,modelName,str(epoch))
    GraphResultsFiles="Results/PartitionResults/{}-{}-GraphEpoch_{}.pkl".format(dataset,modelName,str(epoch))
    with open(GraphResultsFiles,'rb') as f:  
        G=pickle.loads(f.read())
    with open(classiResultsFiles, 'rb') as f:  
        partition=pickle.loads(f.read())
    nx.Graph()
    partitionNew={}
    for key in partition:
        for value in partition[key]:
            partitionNew.update({value: key})
    pos = community_layout(G, partitionNew)
    nx.draw(G, pos, node_color=list(partitionNew.values()))
    #plt.show()
    plt.savefig("GraphPartitionVisualization-{}_{}-{}.png".format(dataset,modelName,epoch))
