timestamp=`date +%Y%m%d%H%M%S`

dataset="Cora"

for i in 16 32
do
 echo "Contraction coefficients $i"
 python3 examples/MyGCN.py  --dataset $dataset --BatchSize $i --ConCoeff 0.6 --Epochs 100  --MonteSize 10 --print_device_useage 'False' --resume 'True' --return_output 'False' --save_recurrence_plots 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done


for i in 32 64 128 256 512 1024
do
 echo "Contraction coefficients $i"
 python3 examples/MyGCN.py  --dataset $dataset --BatchSize $i --ConCoeff 0.6 --Epochs 100  --MonteSize 10 --print_device_useage 'False' --resume 'True' --return_output 'False' --save_recurrence_plots 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done



dataset="ENZYMES"

name='GCN-Contraction_coefficients_changes'

for i in 0.2 0.4 0.6 0.8 0.9 
do
 echo "Contraction coefficients $i"
 python3 examples/MyGCN.py  --dataset $dataset --BatchSize 512 --ConCoeff $i --Epochs 100  --MonteSize 40 --print_device_useage 'False' --resume 'True' --return_output 'False' --save_recurrence_plots 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done


timestamp=`date +%Y%m%d%H%M%S`

for i in 64 128 256 512 1024
do
 echo "Contraction coefficients $i"
 python3 examples/MyGCN.py  --dataset $dataset --BatchSize $i --ConCoeff 0.8 --Epochs 100  --MonteSize 20 --print_device_useage 'False' --resume 'True' --return_output 'False' --save_recurrence_plots 'False' 2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done

dataset="Cora"

