import numpy as np
import matplotlib.pyplot as plt
import os,glob
import seaborn as sns
import matplotlib.style as style 
style.available
sns.set_context('paper')
sns.set()
def PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args):
    Legend=args
    x=np.linspace(start_plot,start_plot+epochs-1,num=epochs).tolist()
    if len(coefficientsFirst)>1:
        coefficients=coefficientsFirst
        parts=round(len(coefficients)/2)
        plt.style.use('seaborn-darkgrid')  
        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[i],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[0])):
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
                plt.plot(x[start_plot:start_plot+epochs],mu[start_plot:start_plot+epochs], lw=1.5)
                
            else:
                plt.plot(x[start_plot:start_plot+epochs],mu[start_plot:start_plot+epochs],'--', lw=1.5)
            
            #plt.fill_between(x[start_plot:start_plot+epochs], (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],alpha=0.5)   
 
            
    elif len(coefficientsSecond)>1:
        coefficients=coefficientsSecond
        parts=round(len(coefficients)/2)
        plt.style.use('seaborn-darkgrid')  

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[i],coefficientsThree[0],coefficientsFour[0])):
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
                plt.plot(x,mu[start_plot:start_plot+epochs], lw=1.5)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', lw=1.5)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],alpha=0.5)  
            
    elif len(coefficientsThree)>1:
        coefficients=coefficientsThree
        parts=round(len(coefficients)/2)
        plt.style.use('seaborn-darkgrid')  

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[0],coefficientsThree[i],coefficientsFour[0])):
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
                plt.plot(x,mu[start_plot:start_plot+epochs], lw=1.5)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', lw=1.5)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],alpha=0.5)   
     
    elif len(coefficientsFour)>1:
        coefficients=coefficientsFour
        parts=round(len(coefficients)/2)
        plt.style.use('seaborn-darkgrid')  

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}*{}*{}*{}*{}*{}*{}*.npy".format(file_constraited,dataset,modelName,coefficientsFirst[0],coefficientsSecond[0],coefficientsThree[0],coefficientsFour[i])):
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
                plt.plot(x,mu[start_plot:start_plot+epochs], lw=1.5)
                
            else:
                plt.plot(x,mu[start_plot:start_plot+epochs],'--', lw=1.5)
            
            plt.fill_between(x, (mu-standard_dev)[start_plot:start_plot+epochs],(mu+standard_dev)[start_plot:start_plot+epochs],alpha=0.5)   

        
    else:
        raise Exception ("Wrong input, please check")
    
    

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(tuple(Legend))
    plt.savefig(save_png_name,dpi=600)

# def PlotMonteCalorsTimesConvergencePth(coefficients,file_path,parts,save_png_name,start_plot):
