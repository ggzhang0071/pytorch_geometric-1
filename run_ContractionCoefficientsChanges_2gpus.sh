timestamp=`date +%Y%m%d%H%M%S`

rm Logs/*.log


dataset="Cora"
name='layersNum_changes'



for i in 4 8 16
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers $i  --ConCoeff 0.95 --CutoffCoeff 2 --num_pre_epochs 200 --num_epochs 200 --MonteSize 10  --resume False 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done


for i in  1 2 4 8 16
do
    python3 ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers $i  --ConCoeff 0.5  --CutoffCoeff 2  --num_epochs 200 --MonteSize 10 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done

<<"COMMENT"
COMMENT