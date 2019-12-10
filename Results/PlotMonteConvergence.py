coefficientsSecond=[0.95]
coefficientsFirst=[1, 2, 4]
dataset="Cora"
file_constraited="TrainConvergence"
save_png_name='LayerNumCompare_{}.png'.format(dataset)
start_plot=0
epochs=200
args=[3, 4,6]

from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args)