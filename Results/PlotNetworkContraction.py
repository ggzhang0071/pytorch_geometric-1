import numpy as np
import matplotlib.pyplot as plt
import os,glob
import seaborn as sns
import matplotlib.style as style 
style.available
sns.set_context('paper')
sns.set()
plt.style.use('seaborn-darkgrid')  

def PlotNetworkContractionNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args):
    Legend=args

    if len(coefficientsFirst)>1:
        coefficients=coefficientsFirst
        parts=round(len(coefficients)/2)
        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[i],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[0])):
                print(file)
                TrainConvergence=np.load(file).tolist()
                TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            mu = np.array(TrainConvergenceAll).mean(axis=0)
            print("contraction coefficients is :{}".format(mu))
            standard_dev = np.array(TrainConvergenceAll).std(axis=0)
            plt.plot(mu, lw=1.5)
            plt.fill_between((mu-standard_dev),(mu+standard_dev),alpha=0.5)  

    elif len(coefficientsSecond)>1:
        coefficients=coefficientsSecond
        parts=round(len(coefficients)/2)

        for i in range(len(coefficients)):
            x=[i for i in range(coefficients[i]+1)]
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[i],coefficientsThree[0],coefficientsFour[0])):
                print(file)
                TrainConvergence=np.load(file,allow_pickle=True).tolist()
                TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            mu = np.array(TrainConvergenceAll).mean(axis=0).tolist()[0]
            standard_dev = np.array(TrainConvergenceAll).std(axis=0).tolist()[0]
            if i<parts:
                plt.plot(x,mu, lw=1.5)
                
            else:
                plt.plot(x,mu,'--', lw=1.5)
            #plt.fill_between((mu-standard_dev),(mu+standard_dev),alpha=0.5)  
            

    plt.xlabel('Depth')
    plt.ylabel('Width')
    plt.legend(tuple(Legend))
    plt.savefig(save_png_name,dpi=600)

#def PlotMonteCalorsTimesConvergencePth(coefficients,file_path,parts,save_png_name,start_plot):
