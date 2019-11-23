coefficientsSecond=[0.2,0.4,0.6,0.8,0.9]
coefficientsFirst=[512]
parts=2
dataset="ENZYMES"
file_constraited="TrainConvergence"
save_png_name='ContractionCoefficientsCompare_{}.png'.format(dataset)
start_plot=30
from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot)