from PlotMonteCalorsConvergence import PlotMonteCalorsTimesConvergenceNpy
coefficientsFirst=[512]
coefficientsSecond=[1]
coefficientsThrees=[[0.1],[0.3],[0.5]]
coefficientsFour=[0.0,0.01]
dataset="Pubmed"
modelName="ChebConvNet"
file_constraited="Results/{}Convergence/AlgebraicConectivityTestConvergence".format(dataset)
save_png_name='Results/RegularizationCoeffiCompare_{}.png'.format(dataset)
start_plot=60
epochs=130
args=coefficientsFour
for coefficientsThree in coefficientsThrees:
    PlotMonteCalorsTimesConvergenceNpy(dataset,modelName,file_constraited,coefficientsFirst,coefficientsSecond,coefficientsThree,coefficientsFour,save_png_name,start_plot,epochs,*args)