coefficientsFirst=[128]
coefficientsSecond=[1,2, 3, 4]
coefficientsThree=[0.8]
coefficientsFour=[0.1]
modelName="ChebConvNet"
Prefix="NewNetworkSizeAdjust"
dataset="Cora"
FileName=dataset+"Convergence/"
file_constraited=FileName+Prefix
save_png_name='NewNetworkSizeAdjustCompare_{}-{}.png'.format(dataset,modelName)
start_plot=0
epochs=200
args=coefficientsSecond
from PlotNetworkContraction import PlotNetworkContractionNpy
PlotNetworkContractionNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)
