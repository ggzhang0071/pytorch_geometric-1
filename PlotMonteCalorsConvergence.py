import numpy as np
import matplotlib.pyplot as plt
import os,glob
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import pdb

"""
print(mpl.rcParmas)
mpl.rcParmas['font.family']="Times New Roman"
mpl.rcParmas['lines.width']=2
mpl.rcParmas['figure.figsize']=8,6"""

colors=[(248/255,25/255,25/255),(40/255,172/255,82/255),(161/255,80/255,159/255),(0/255,127/255,182/255)]
#colors=["#fc8d59","#ffffbf","#91cf60"]
camp=ListedColormap(colors)
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 60}

mpl.rc('font', **font)



def PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,
                                       coefficientsSecond,coefficientsThree,coefficientsFour,start_plot,epochs,*args):
    Legend=args
    x=np.linspace(start_plot,start_plot+epochs-1,num=epochs).tolist()
    if len(coefficientsFirst)>1:
        coefficients=coefficientsFirst
        parts=round(len(coefficients)/2)
        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            FileNames="{}-{}*{}*{}_{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[i],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[0])
            print(FileNames)
            for file in glob.glob(FileNames):
                print(file)
                TrainConvergence=np.load(file).tolist()
                if max(TrainConvergence)>20:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=20:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            mu = np.array(TrainConvergenceAll).mean(axis=0)
            standard_dev = np.array(TrainConvergenceAll).std(axis=0)
            
            if i<parts:
                plt.plot(x,mu[start_plot:start_plot+epochs], c=colors[i],lw=2)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', c=colors[i],lw=2)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],facecolor=colors[i],alpha=0.5)   
 
    elif len(coefficientsSecond)>1:
        coefficients=coefficientsSecond
        parts=round(len(coefficients)/2)

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            FileNames="{}-{}*{}*{}_{}_{}_{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[i],coefficientsThree[0],coefficientsFour[0])
            print(FileNames)
            for file in glob.glob(FileNames):
                print(file)
                TrainConvergence=np.load(file).tolist()
                if max(TrainConvergence)>20000:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=4040:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            mu = np.array(TrainConvergenceAll).mean(axis=0)
            standard_dev = np.array(TrainConvergenceAll).std(axis=0)
            if i<parts:
                plt.plot(x,mu[start_plot:start_plot+epochs], c=colors[i],lw=1.5)
                
            elif i>=parts :
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', c=colors[i],lw=1.5)
            
            #plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],facecolor=colors[i],alpha=0.5)  
            
    elif len(coefficientsThree)>1:
        coefficients=coefficientsThree
        parts=round(len(coefficients)/2)

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            FileFolder="{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[0],coefficientsThree[i],coefficientsFour[0],1)
            print(FileFolder)
            for file in glob.glob(FileFolder):
                print(file)
                TrainConvergence=np.load(file).tolist()
                if max(TrainConvergence)>20:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=20:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            mu = np.array(TrainConvergenceAll).mean(axis=0)
            standard_dev = np.array(TrainConvergenceAll).std(axis=0)
            if mu.any():
                continue
            if i<parts:
                plt.plot(x,mu[start_plot:start_plot+epochs], c=colors[i],lw=2)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', lc=colors[i],lw=2)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],facecolor=colors[i],alpha=0.5)   
     
    elif len(coefficientsFour)>1:
        coefficients=coefficientsFour
        parts=round(len(coefficients)/2)
        #pdb.set_trace()
        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            FileFolder="{}*{}*{}*{}*{}*{}_{}-*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[i])
            print(FileFolder)
            for file in glob.glob(FileFolder):
                print(file)
                N=200
                TrainConvergenceTmp=np.load(file).tolist()
                TrainConvergence=[0]*(N-len(TrainConvergenceTmp))
                TrainConvergence+=TrainConvergenceTmp
                if max(TrainConvergence)>20:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=20:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))   
            if len(TrainConvergenceAll)==0:
                 raise Exception("Input file isn't exists")
            mu = np.array(TrainConvergenceAll).mean(axis=0)
            standard_dev = np.array(TrainConvergenceAll).std(axis=0)
            saveMeanStd="{}MeanAndStd-{}-{}-{}-{}-{}-{}.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[i])
            np.save(saveMeanStd,[mu,standard_dev])

            #pdb.set_trace()
            if i<parts:
                plt.plot(x,mu[start_plot:start_plot+epochs], c=colors[i],lw=2)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--',c=colors[i],lw=2)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],facecolor=colors[i],alpha=0.5)   

        
    else:
        raise Exception ("Wrong input, please check")
    plt.style.use(['science','seaborn-white','no-latex'])
    #plt.legend(tuple(Legend))

# def PlotMonteCalorsTimesConvergencePth(coefficients,file_path,parts,start_plot):
