dataset="Cora"

name='Contraction_coefficients_changes'

for i in 0.1 0.2 0.4 0.8
do
 echo "Contraction coefficients $i"
 python3 MyGCN.py  --ConCoeff $i --Epochs 10  2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done

