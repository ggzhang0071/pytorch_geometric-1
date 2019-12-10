import numpy as np
import matplotlib.pyplot as plt
import os,glob

def PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args):
    Legend=args
    x=np.linspace(start_plot,start_plot+epochs-1,num=epochs).tolist()
    if len(coefficientsFirst)>1:
        coefficients=coefficientsFirst
        parts=round(len(coefficients)/2)
        plt.style.use('ggplot')  

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}-{}*{}_{}*.npy".format(file_constraited,dataset,coefficientsFirst[i],coefficientsSecond[0])):
                print(file)
                TrainConvergence=np.load(file)
                if max(TrainConvergence)>5:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=5:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            TestConvergenceAvg=[sum(x)/len(TrainConvergenceAll) for x in zip(*TrainConvergenceAll)]
            TrainLossEpoches=TestConvergenceAvg
            if i<parts:
                plt.plot(x,TrainLossEpoches[start_plot:start_plot+epochs], lw=1.5)
            else:
                plt.plot(x,TrainLossEpoches[start_plot:start_plot+epochs],'--', lw=1.5)
            
    elif len(coefficientsSecond)>1:
        coefficients=coefficientsSecond
        parts=round(len(coefficients)/2)+1
        plt.style.use('ggplot')  

        for i in range(len(coefficients)):
            TrainConvergenceAll=[]
            for file in glob.glob("{}-{}*{}_{}*.npy".format(file_constraited,dataset,coefficientsFirst[0],coefficientsSecond[i])):
                print(file)
                TrainConvergence=np.load(file)
                if max(TrainConvergence)>20:
                    print("{} maximum is:{}".format(file,max(TrainConvergence)))
                    os.remove(file)
                if len(TrainConvergence)>40 and max(TrainConvergence)<=20:
                    TrainConvergenceAll.append(TrainConvergence)
            print("coefficient of {} num is: {}".format(coefficients[i],len(TrainConvergenceAll)))            
            TestConvergenceAvg=[sum(x)/len(TrainConvergenceAll) for x in zip(*TrainConvergenceAll)]
            TrainLossEpoches=TestConvergenceAvg
            if i<parts:
                plt.plot(x,TrainLossEpoches[start_plot:start_plot+epochs], lw=1.5)
            else:
                plt.plot(x,TrainLossEpoches[start_plot:start_plot+epochs],'--', lw=1.5)
        
    else:
        raise Exception ("Wrong input, please check")
    
    

    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.legend(tuple(Legend))
    plt.savefig(save_png_name,dpi=600)

#def PlotMonteCalorsTimesConvergencePth(coefficients,file_path,parts,save_png_name,start_plot):
