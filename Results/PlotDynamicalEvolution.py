import numpy as np
import glob
import matplotlib.pyplot as plt
FileName="DiagElement/"
dataset="Cora"
modelName="GCN"
FileConstrant="Pretrain"
save_png_name="{}*{}*{}*{}.png".format(FileName,dataset,modelName,FileConstrant)
for file in glob.glob("{}*{}*{}*{}.npy".format(FileName,dataset,modelName,FileConstrant)):
    if not file:
        EvolutionDynamics=np.load("./DiagElement/Cora-GCN-Pretrain-DiagElement.npy")
        print(EvolutionDynamics)
    else:
        print("wrong file")


for i in range(EvolutionDynamics.shape[1]):
    plt.plot([1,2,3,4],EvolutionDynamics.T[i])
plt.legend([1,2,3,4,5])
plt.xlabel("layer")
plt.ylabel("")

plt.savefig(save_png_name,dpi=600)