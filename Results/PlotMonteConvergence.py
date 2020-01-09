from SeabonPlotConvergence import PlotMonteCalorsTimesConvergenceNpySeaborn

coefficientsSecond=[0.2,0.4,0.6,0.8,0.9]
coefficientsFirst=[512]
dataset="ENZYMES"

file_constraited="ENZYMESConvergence/TrainConvergence"
save_png_name='ContractionCoefficientsCompare_{}.png'.format(dataset)
start_plot=30
epochs=70
args=coefficientsSecond
PlotMonteCalorsTimesConvergenceNpySeaborn(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args)


"""from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy

coefficientsSecond=[0.2,0.4,0.6,0.8,0.9]
coefficientsFirst=[512]
dataset="ENZYMES"

file_constraited="ENZYMESConvergence/TrainConvergence"
save_png_name='ContractionCoefficientsCompare_{}.png'.format(dataset)
start_plot=30
epochs=70
args=coefficientsSecond
PlotMonteCalorsTimesConvergenceNpy(dataset,file_constraited,coefficientsFirst,coefficientsSecond,save_png_name,start_plot,epochs,*args)"""