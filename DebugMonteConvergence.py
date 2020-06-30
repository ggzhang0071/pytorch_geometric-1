from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy
coefficientsFirst=[512]
coefficientsSecond=[1]
coefficientsThree=[0.3]
coefficientsFour=[0.0,0.001,0.01]
dataset="Cora"
modelName="GCN"
file_constraited="Results/{}Convergence/AlgebraicConectivityTestConvergence".format(dataset)
save_png_name='Results/RegularizationCoeffiCompare_{}.png'.format(dataset)
start_plot=60
epochs=80
args=coefficientsFour
PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)
