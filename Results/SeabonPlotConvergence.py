import numpy as np
import matplotlib.pyplot as plt
import os,glob
import seaborn as sns
import pandas as pd


def PlotMonteCalorsTimesConvergenceNpySeaborn(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args):
    Legend=args
    #plt.figure(figsize = (16,9)) # figure size with ratio 16:9
    sns.set(style='darkgrid',) # background darkgrid style of graph 
    sns.color_palette("dark", 10)

    #sns.set(style="ticks")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
   
    x=np.linspace(start_plot,start_plot+epochs-1,num=epochs).tolist()
    if len(coefficientsFirst)>1:
        coefficients=coefficientsFirst
        parts=round(len(coefficients)/2)

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}-{}*{}_{}*.npy".format(file_constraited,dataset,coefficientsFirst[i],coefficientsSecond[0])):
                print(file)
                TrainConvergence=np.load(file).tolist()
                if max(TrainConvergence)>5:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                elif len(TrainConvergence)>40:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))     
            df = pd.DataFrame(TrainConvergenceAll,index=range(0,len(TrainConvergenceAll)))
            ax = sns.tsplot(data=df.values) #, err_style="unit_traces")
            mu = df.mean(axis=0)
            standard_dev = df.std(axis=0)
            #ax.errorbar(x, mu, yerr=standard_dev, fmt='-o') #fmt=None to plot bars only
            
            if i<parts:
                sns.lineplot(x=x,y=mu[start_plot:start_plot+epochs])
                
            else:
                sns.lineplot(x=x,y=mu[start_plot:start_plot+epochs])
            
            
    elif len(coefficientsSecond)>1:
        coefficients=coefficientsSecond
        parts=round(len(coefficients)/2)

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}-{}*{}_{}*.npy".format(file_constraited,dataset,coefficientsFirst[0],coefficientsSecond[i])):
                print(file)
                TrainConvergence=np.load(file).tolist()
                if max(TrainConvergence)>5:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                elif len(TrainConvergence)>40:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            df = pd.DataFrame(TrainConvergenceAll,index=range(0,len(TrainConvergenceAll)))
            ax = sns.lineplot(x=x,data=df.values) #, err_style="unit_traces")
            mu = df.mean(axis=1)
            standard_dev  = df.std(axis=1)
            #ax.errorbar(x, mu, yerr=standard_dev, fmt='-o') #fmt=None to plot bars only
            
            if i<parts:
                sns.lineplot(x=x,y=mu[start_plot:start_plot+epochs])
                
            else:
                sns.lineplot(x=x,y=mu[start_plot:start_plot+epochs])
        
    else:
        raise Exception ("Wrong input, please check")
    
    

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(tuple(Legend))
    plt.savefig(save_png_name,dpi=600)

#def PlotMonteCalorsTimesConvergencePth(coefficients,file_path,parts,save_png_name,start_plot):
