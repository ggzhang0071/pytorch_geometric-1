timestamp=`date +%Y%m%d%H%M%S`

rm Logs/*.log 
#'MUTAG' 'ENZYMES' 'Citeseer' 
for dataset in   'Cora' 
do
for  modelName in  'GCN'  
do
for NumLayers in  1
do
for LR in 0.5
do
for BatchSize in 512
do
#
for regularization_coef in 0 2 4 6 8
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize $BatchSize --NumLayers $NumLayers --regularization_coef $regularization_coef --ConCoeff 0.99 --num_pre_epochs 100 --num_epochs 200 --MonteSize 1 --LR $LR --modelName $modelName --PruningTimes 1 --resume False  2>&1 |tee Logs/${modelName}_${dataset}_${regularization_coef}_$timestamp.log


done
done
done
done
done
done


<<"COMMENT"
#'Citeseer' 'Pubmed' 'Cora' 'Reddit'  'Amazon'
for dataset in  'CoraFull' 
do
for NumLayers in 1 2 3 4 6
do
    python3 ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $NumLayers  --ConCoeff 0.99 --CutoffCoeff 2 --num_pre_epochs 60 --num_epochs 200 --MonteSize 10 --LR 0.2  --modelName $modelName --PruningTimes 1 --resume True 2>&1 |tee Logs/${modelName}_${dataset}_$timestamp.log

done
modelName='ChebConvNet'

for dataset in  'Cora' 'Citeseer' 'Pubmed'
do
for NumLayers in 1 2 3 4 
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $NumLayers  --ConCoeff 0.8 --CutoffCoeff 2 --num_pre_epochs 120 --num_epochs 200 --MonteSize 10 --LR 0.1  --modelName $modelName --PruningTimes 2 --resume True 2>&1 |tee Logs/${modelName}_${dataset}_$timestamp.log

done


for dataset in  'Cora' 'Citeseer' 'Pubmed'
do
for lr in 0.001 0.01 0.1 0.5
do
    python3  ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $NumLayers  --ConCoeff 0.8 --CutoffCoeff 2 --num_pre_epochs 120 --num_epochs 200 --MonteSize 10 --LR $lr  --modelName $modelName --PruningTimes 2 --resume True 2>&1 |tee Logs/${modelName}_${dataset}_$timestamp.log

done
done



for i in  1 2 4 6 
do
    python3   examples/ConvexPruning.py --dataset $dataset --BatchSize 128 --NumLayers $i  --ConCoeff 0.8 --CutoffCoeff 2 --num_pre_epochs 150 --num_epochs 150 --MonteSize 10 --LR 0.05 --resume False 2>&1 |tee Logs/${name}_${dataset}_$timestamp.log
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
