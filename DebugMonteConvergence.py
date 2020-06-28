from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy
coefficientsFirst=[512]
coefficientsSecond=[1]
coefficientsThree=[0.1]
coefficientsFour=[1.0,2.0]
dataset="Cora"
modelName="GCN"
file_constraited="Results/{}Convergence/AlgebraicConectivityTestConvergence".format(dataset)
save_png_name='RegularizationCoeffiCompare_{}.png'.format(dataset)
start_plot=100
epochs=100
args=coefficientsFour
PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)

