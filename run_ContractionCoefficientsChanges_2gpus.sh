dataset="Cora"

name='Contraction_coefficients_changes'

for i in 0.1 0.2
do
 echo "Contraction coefficients $i"
 python3 Mygcn.py  --ConCoeff $i --Epochs 20  2>&1 |tee Logs/${name}_${dataset}_${i}_$timestamp.log
done

