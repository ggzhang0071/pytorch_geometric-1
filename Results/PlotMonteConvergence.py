coefficientsThree=[0.8]
coefficientsFirst=[128]
coefficientsSecond=[2]
coefficientsFour=[0.001, 0.01, 0.1, 0.5]
dataset="Pubmed"
modelName="GCN"
Prefix="TrainConvergence"
FileName=dataset+"Convergence/"
file_constraited=FileName+Prefix
save_png_name='LayerNumCompare_{}.png'.format(dataset)
start_plot=0
epochs=200
args=coefficientsFour
from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)