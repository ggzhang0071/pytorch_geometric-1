timestamp=`date +%Y%m%d%H%M%S`

rm Logs/*.log


dataset="Cora"
name='layersNum_changes'

for i in 1 2 3 4 5 6
do
    python3 ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers $i  --ConCoeff 0.95 --CutoffCoeff 2  --num_epochs 200 --MonteSize 20 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done



for i in  1 2 3 4 5 6
do
    python3 ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers $i  --ConCoeff 0.5  --CutoffCoeff 2  --num_epochs 200 --MonteSize 20 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done
