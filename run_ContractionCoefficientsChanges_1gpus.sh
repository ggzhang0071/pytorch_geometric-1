timestamp=`date +%Y%m%d%H%M%S`

rm Logs/*.log 
#'MUTAG' 'ENZYMES' 
for dataset in 'Pubmed' 'Cora'  
do
for  modelName in   'ChebConvNet' 
do
for NumLayers in 1
do
for LR in 0.5
do
for BatchSize in 512
do
for StartTopoCoeffi  in  0.1  
do 
for VectorPairs in 1 
do
for WeightCorrectionCoeffi in  0.001 0.01 0.1 1
do
    python3  ConvexPruning.py --dataset $dataset --modelName $modelName --BatchSize $BatchSize --NumLayers $NumLayers --VectorPairs $VectorPairs --StartTopoCoeffi $StartTopoCoeffi --ConCoeff 0.95 --num_pre_epochs 100 --num_epochs 200 --MonteSize 1 --LR $LR  --WeightCorrectionCoeffi $WeightCorrectionCoeffi --PruningTimes 1 --resume False  2>&1 |tee Logs/${modelName}_${dataset}_${StartTopoCoeffi}-${WeightCorrectionCoeffi}_${VectorPairs}_$timestamp.log


done
done
done
done
done
done
done
done
