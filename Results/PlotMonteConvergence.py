coefficientsSecond=[4, 8, 16, 32, 64]
coefficientsFirst=[16]
dataset="Cora"
file_constraited="TrainConvergence"
save_png_name='ContractionCoefficientsCompare_{}.png'.format(dataset)
start_plot=30
epochs=70
from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs)