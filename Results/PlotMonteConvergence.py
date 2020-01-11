"""coefficientsSecond=[0.8]
coefficientsFirst=[1,2,4,6]
dataset="ENZYMES"
file_constraited="TrainConvergence"
save_png_name='LayerNumCompare_{}.png'.format(dataset)
start_plot=0
epochs=150
args=[3,4,6,8]
from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args)"""

coefficientsSecond=[0.001, 0.01, 0.1, 0.5]
coefficientsFirst=[128]
dataset="Cora"
modelName="GCN"
Prefix="TrainConvergence"
FileName=dataset+"Convergence/"
file_constraited=FileName+Prefix
save_png_name='LayerNumCompare_{}.png'.format(dataset)
start_plot=0
epochs=200
args=coefficientsSecond
from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args)