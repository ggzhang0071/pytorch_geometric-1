from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy
coefficientsFirst=[512]
coefficientsSecond=[1]
coefficientsThree=[0.99]
coefficientsFour=[0.0, 0.001, 0.01, 0.1, 1]
dataset="Cora"
modelName="GCN"
file_constraited="Results/{}Convergence/AlgebraicConectivityTrainConvergence".format(dataset)
save_png_name='RegularizationCoeffiCompare_{}.png'.format(dataset)
start_plot=100
epochs=200
args=coefficientsFour
PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)


