timestamp=`date +%Y%m%d%H%M%S`

rm Logs/*.log


modelName='GCN'

#
for dataset in  'Cora' 'Citeseer' 'Pubmed'
do
for NumLayers in 1 2  3 4 6 8 
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $NumLayers  --ConCoeff 0.8 --CutoffCoeff 2 --num_pre_epochs 100 --num_epochs 200 --MonteSize 10 --lr 0.1  --modelName $modelName --PruningTimes 2 --resume True 2>&1 |tee Logs/${modelName}_${dataset}_$timestamp.log
    
done
done



<<"COMMENT"
for i in  1 2 4 6 
do
    python3   examples/ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $i  --ConCoeff 0.8 --CutoffCoeff 2 --num_pre_epochs 150 --num_epochs 150 --MonteSize 10 --lr 0.05 --resume False 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done

for i in  1 2 4 8 
do
    python3   examples/ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers $i  --ConCoeff 0.5  --CutoffCoeff 2  --num_epochs 30 --MonteSize 10 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done


for i in 1.1  1.5 2
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize 512 --NumLayers 1 --ConCoeff 0.95 --CutoffCoeff $i --num_pre_epochs 200 --num_epochs 200 --MonteSize 10  --resume True 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
done

COMMENT