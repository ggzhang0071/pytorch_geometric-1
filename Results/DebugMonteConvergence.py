coefficientsFirst=[512,1024]
coefficientsSecond=[2]
coefficientsThree=[0.99]
coefficientsFour=[0.1]
modelName="ChebConvNet"
Prefix="NewNetworkSizeAdjust"
dataset="ENZYMES"
FileName=dataset+"Convergence/"
file_constraited=FileName+Prefix
save_png_name='NewNetworkSizeAdjustCompare_{}-{}.png'.format(dataset,modelName)
start_plot=0
epochs=200
args=[3,4,5]
from PlotNetworkContraction import PlotNetworkContractionNpy
PlotNetworkContractionNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)