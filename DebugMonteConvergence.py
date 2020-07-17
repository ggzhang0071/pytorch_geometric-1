from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy
coefficientsFirst=[512,10]
coefficientsSecond=[0.001]
coefficientsThree=[0.1]
coefficientsFour=[1]
dataset="Pubmed"
modelName="GCN"
file_constraited="Results/{}Convergence/AlgebraicConectivityTestConvergence".format(dataset)
save_png_name='Results/WeightCorrectionCoeffiCompare-{}-{}.png'.format(dataset,modelName)
start_plot=30
epochs=20
args=coefficientsThree
PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)